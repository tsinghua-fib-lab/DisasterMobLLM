import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
from utils import *
from sklearn.cluster import KMeans
import torch.nn.functional as F
import joblib
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    LlamaModel,
    LlamaForCausalLM,
)


class TrajectoryTokenizor(nn.Module):
    def __init__(self, dataset, traj_pad_length, path):
        super(TrajectoryTokenizor, self).__init__()
        self.dataset = dataset
        self.path = path
        self.pid2idx = self.build_pid2idx_list()
        self.traj_pad_length = traj_pad_length

    def build_pid2idx_list(self):
        pid2idx = {}
        index = 1
        for dataset in self.dataset:
            city, mode = dataset.split("-")
            with open(f"{self.path}/{city}/data_{mode}.pk", "rb") as f:
                data = pickle.load(f)
            for vid in data["vid_list"]:
                if not (vid == "unk") and (not vid in pid2idx.keys()):
                    pid2idx[vid] = index
                    index += 1
        return pid2idx

    def forward(self, trajs):
        traj_rec = []
        for point in trajs:
            traj_rec.append(self.pid2idx[point] if point in self.pid2idx.keys() else 0)
        padded_trajectory = np.pad(
            traj_rec, (0, self.traj_pad_length - len(traj_rec)), "constant"
        )
        return padded_trajectory


# 轨迹编码器
class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num):
        super(TrajectoryEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=2
        )
        self.decoder = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, class_num)

    def forward(self, trajectory):
        # 投影到隐藏维度 (batch_size, seq_length, hidden_dim)
        trajectory = self.fc(trajectory)

        # 将轨迹转置为 (seq_length, batch_size, hidden_dim)
        trajectory = trajectory.permute(1, 0, 2)
        out = self.transformer_encoder(trajectory).mean(dim=0)
        trajectory_embedding = self.decoder(out)
        logits = self.classifier(trajectory_embedding)
        logits = F.log_softmax(logits, dim=1)
        # return trajectory_embedding.mean(dim=0)  # 平均池化
        return trajectory_embedding, logits


class ImmobilityEmbedding(nn.Module):
    def __init__(self, tokenizer, llm_model):
        super(ImmobilityEmbedding, self).__init__()
        token = tokenizer.encode_plus(
            "stay still",
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=False,
            return_attention_mask=False,
            return_tensors="pt",
        )
        immob_embeddings = (
            llm_model.get_input_embeddings()(token["input_ids"]).detach()
            if isinstance(llm_model, LlamaForCausalLM)
            else llm_model.embed_tokens(token["input_ids"]).detach()
        )
        self.immob_embeddings = torch.mean(immob_embeddings, dim=1).requires_grad_(True)

    def forward(self):
        return self.immob_embeddings


# CLIP模型
class CLIPModel(nn.Module):
    def __init__(
        self,
        dataset,
        path,
        traj_pad_length,
        input_dim,
        hidden_dim,
        proto_dim,
        attn_dim,
        intention_dim,
        text_vocab,
        use_LLM,
        llm_model,
        class_num,
    ):
        super(CLIPModel, self).__init__()
        self.dataset = dataset
        self.traj_pad_length = traj_pad_length
        self.traj_tokenizor = TrajectoryTokenizor(
            dataset=dataset, traj_pad_length=traj_pad_length, path=path
        )
        self.trajectory_encoder = TrajectoryEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            class_num=class_num,
        )
        self.text_tokenizor = AutoTokenizer.from_pretrained(use_LLM)
        self.immob_emb = nn.Parameter(torch.randn(1, intention_dim))
        self.text_vocab = text_vocab
        vocab_dim = self.text_vocab.shape[0]  # 词表的每个词的dim
        self.vocab_to_proto = nn.Linear(vocab_dim, proto_dim)
        self.Wq = nn.Linear(intention_dim, attn_dim)
        self.Wk = nn.Linear(hidden_dim, attn_dim)
        self.Wv = nn.Linear(hidden_dim, attn_dim)
        self.Attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=8)
        self.fc = nn.Linear(attn_dim, hidden_dim)

    def forward(self, trajectory, intention):
        trajectory_embedding, logits = self.trajectory_encoder(
            trajectory
        )  # batch_size,text_dim
        # text_embedding = self.text_encoder(text_input_ids, text_attention_mask)
        mask = intention.sum(dim=1) == 0  # 找出所有全为0的行
        if mask.any():  # 如果有全为0的行
            # 用 emb 替换那些全为0的行
            intention[mask] = self.immob_emb.expand(
                mask.sum().item(), -1
            )  # -1 表示保持原维度
        prototype = self.vocab_to_proto(self.text_vocab.T).T
        q = self.Wq(intention)
        k = self.Wk(prototype)
        v = self.Wv(prototype)
        x = self.Attn(q, k, v)[0]
        x = F.relu(x)
        intention_embedding = self.fc(x)
        return trajectory_embedding, intention_embedding, logits

    # def get_intention(self, trajectory):
    #     trajectory_embedding = self.trajectory_encoder(trajectory).cpu()
    #     max_similarity = -np.inf
    #     for intention in self.intention_embeddings:
    #         similarity = nn.functional.cosine_similarity(
    #             trajectory_embedding, self.intention_embeddings[intention]
    #         )
    #         if similarity > max_similarity:
    #             max_similarity = similarity
    #             max_similar_intention = intention
    #     return max_similar_intention


class CLIPDataset(Dataset):
    def __init__(self, datasets, traj_pad_length, cluster, path, city_immob_indexs={}):
        self.datasets = datasets
        self.traj_pad_length = traj_pad_length
        self.cluster = cluster
        self.path = path
        datas = []
        intention_dis = {"immob": 0}
        for dataset in datasets:
            city, mode = dataset.split("-")
            if not city in city_immob_indexs.keys():
                city_immob_indexs[city] = 0
            not_immob_index = city_immob_indexs[city]
            with open(
                f"{path}/util_datas/CLIP_input/{city}.json",
                "r",
            ) as f:
                tca_data = json.load(f)
            with open(
                f"{path}/util_datas/CLIP_dataset/{city}/{mode}.json",
                "r",
            ) as f:
                input_data = json.load(f)
            for session in input_data:
                data = []
                for index in range(len(session["vectors"])):
                    vector = session["vectors"][index] + session["time_emb"][index]
                    if not session["vectors"][index][64] == 0:
                        intention_emb = tca_data[not_immob_index]
                        intention, intention_index = self.cluster.get_cluster(
                            intention_emb
                        )
                        not_immob_index += 1
                        # else:
                        #     intention = np.zeros(20).tolist()
                        #     intention_index = -1
                        if not intention_index in intention_dis.keys():
                            intention_dis[intention_index] = 0
                        else:
                            intention_dis[intention_index] += 1
                        data.append(
                            {
                                "switch_feature": vector,
                                "intention": intention,
                                "intention_index": intention_index,
                            }
                        )
                    else:
                        data.append(
                            {
                                "switch_feature": vector,
                                "intention": np.zeros(20).tolist(),
                                "intention_index": -1,
                            }
                        )
                        intention_dis["immob"] += 1
                datas.extend(data)
            city_immob_indexs[city] = not_immob_index
        print("intention distribution:", intention_dis)
        self.datas = datas
        self.immob = len(intention_dis) - 1
        for dindex, data in enumerate(self.datas):
            if data["intention_index"] - 1:
                self.datas[dindex]["intention_index"] = self.immob
        self.city_immob_indexs = city_immob_indexs

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        # trajectory = np.array([x["switch_feature"] for x in data[:-1]])
        trajectory = np.array(data["switch_feature"])
        intention = np.array(data["intention"])
        intention_index = np.array(data["intention_index"])
        # if trajectory.shape[0] > self.traj_pad_length:
        #     padded_trajectory = trajectory[: self.traj_pad_length, :]
        # else:
        #     padded_trajectory = np.pad(
        #         trajectory,
        #         ((0, self.traj_pad_length - trajectory.shape[0]), (0, 0)),
        #         mode="constant",
        #         constant_values=-1,
        #     )
        # return padded_trajectory, intention, intention_index
        return trajectory, intention, intention_index


class Cluster:
    def __init__(self, datasets, max_cluster_num, path, have_c=False):
        self.datasets = datasets
        self.max_cluster_num = max_cluster_num
        self.modes = ["before", "dis", "after"]
        self.have_c = have_c
        if have_c:
            kmeans = joblib.load(f"{path}/util_datas/kmeans.pkl")
            self.model = kmeans
            with open(
                f"{path}/util_datas/cluster_emb.json",
                "r",
            ) as f:
                self.cluster_emb = json.load(f)
        else:
            vectors = []
            for dataset in datasets:
                with open(
                    f"{path}/util_datas/CLIP_input/{dataset}.json",
                    "r",
                ) as f:
                    data = json.load(f)
                    vectors.extend(data)

            n_components = 15  # 你可以根据需要调整这个值
            kmeans = KMeans(n_clusters=n_components, random_state=42)
            kmeans.fit(vectors)
            self.model = kmeans
            labels = kmeans.labels_
            joblib.dump(
                kmeans,
                f"{path}/util_datas/kmeans.pkl",
            )

            # 获取聚类中心
            centers = kmeans.cluster_centers_
            cluster_means = {}
            for i in range(n_components):
                # 计算这些数据点的平均值
                mean_values = centers[i]

                # 将平均值添加到列表
                cluster_means[i] = mean_values.tolist()
            with open(
                f"{path}/util_datas/cluster_emb.json",
                "w",
            ) as f:
                json.dump(cluster_means, f)
            self.cluster_emb = cluster_means

    def get_cluster(self, traj):
        cluster = self.model.predict(np.array(traj).reshape(1, -1))
        emb = self.cluster_emb[str(cluster[0])]
        return emb, cluster[0]


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, traj_embeddings, text_embeddings):
        traj_embeddings = F.normalize(traj_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # 计算相似度矩阵
        sim_matrix = (
            torch.matmul(traj_embeddings, text_embeddings.t()) / self.temperature
        )
        batch_size = traj_embeddings.size(0)
        labels = torch.arange(batch_size).to(sim_matrix.device)
        loss_i = F.cross_entropy(sim_matrix, labels)
        loss_t = F.cross_entropy(sim_matrix.t(), labels)
        loss = (loss_i + loss_t) / 2.0

        return loss


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha  # InfoNceLoss的权重
        self.beta = beta  # CrossEntropy的权重
        self.info_nce_loss = InfoNCELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(
        self, trajectory_embedding, intention_embedding, intention_pred, intention_index
    ):

        info_nce_loss = self.info_nce_loss(trajectory_embedding, intention_embedding)
        cross_entropy = self.cross_entropy_loss(intention_pred, intention_index)

        # 复合损失
        loss = self.alpha * info_nce_loss + self.beta * cross_entropy
        return loss, info_nce_loss, cross_entropy

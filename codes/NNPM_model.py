from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
from main_model import *
import geopandas as gpd

# class NNPMDataset(Dataset):
#     def __init__(
#         self,
#         dataset,
#         traj_encoder,
#         device,
#         hidden_dim,
#         cluster_emb,
#         clip_model,
#         traj_pad_length,
#     ):
#         self.dataset = dataset
#         self.hidden_dim = hidden_dim
#         self.path = "/your_path/Mob_data/datas"
#         self.trajectories = []
#         self.clip_model = clip_model
#         self.traj_encoder = traj_encoder
#         self.cluster_emb = {}
#         self.traj_pad_length = traj_pad_length
#         if os.path.exists(f"{self.path}/util_datas/main_dataset/{dataset}/before.json"):
#             with open(
#                 f"{self.path}/util_datas/main_dataset/{dataset}/before.json",
#                 "r",
#             ) as f:
#                 data = json.load(f)
#             self.trajectories.extend(data)
#         grids = []
#         for info in self.trajectories:
#             grids.extend(info["grid_seq"])
#         self.dataset_index = int(grids[0] / 1e8)
#         self.grids = list(range(int(max(grids) - self.dataset_index * 1e8) + 1))
#         emb = torch.tensor(
#             np.array(list(cluster_emb.values())),
#             dtype=torch.float32,
#             requires_grad=False,
#         ).to(device)
#         fake_traj = torch.tensor(
#             np.zeros((1, 1, 111)), dtype=torch.float32, requires_grad=False
#         ).to(device)
#         _, intention_embedding = self.clip_model(fake_traj, emb)
#         intention_embedding = intention_embedding.detach().cpu().numpy().tolist()
#         for index in range(len(intention_embedding)):
#             self.cluster_emb[index] = intention_embedding[index]
#         self.cluster_emb[-1] = np.zeros(self.hidden_dim).tolist()

#     def __len__(self):
#         return len(self.trajectories)

#     def __getitem__(self, idx):
#         traj_info = self.trajectories[idx]
#         reg_seq = traj_info["grid_seq"]
#         time_emb_seq = [np.array(x)[94:102].tolist() for x in traj_info["st_emb_seq"]]
#         time_emb_seq.append(np.array(traj_info["st_emb_seq"][-1])[102:110].tolist())
#         intention_embedding = [
#             np.array(x)[0:66].tolist() for x in traj_info["st_emb_seq"]
#         ]
#         [intention_emb_seq, _] = self.get_intention_and_emb_seq(traj_info)
#         return_reg_seq = np.zeros(
#             (self.traj_pad_length, len(self.grids)), dtype=np.float32
#         )
#         for j in range(len(reg_seq) - 1):
#             return_reg_seq[j, int(reg_seq[j] - self.dataset_index * 1e8)] = 1.0

#         return_int_seq = np.pad(
#             np.array(intention_embedding[:-1]),
#             ((0, self.traj_pad_length - len(intention_embedding) + 1), (0, 0)),
#             "constant",
#         )
#         return_time_seq = np.pad(
#             time_emb_seq[:-1],
#             ((0, self.traj_pad_length - len(time_emb_seq) + 1), (0, 0)),
#             "constant",
#         )
#         return (
#             return_reg_seq,
#             return_int_seq,
#             return_time_seq,
#             np.array(reg_seq[-1]) - self.dataset_index * 1e8,
#         )

#     def get_intention_and_emb_seq(self, traj_info):
#         intention_emb_seq = []
#         for point in traj_info["st_emb_seq"]:
#             encoded_input = np.expand_dims(point, axis=0)
#             encoded_input = np.expand_dims(encoded_input, axis=0)
#             if not point[64] == 0:
#                 with torch.no_grad():
#                     model_output = self.traj_encoder(encoded_input)
#                 intention_emb_seq.append(model_output.cpu().numpy().tolist()[0])
#             else:
#                 intention_emb_seq.append(np.zeros(self.hidden_dim).tolist())
#         cluster_emb = np.array(list(self.cluster_emb.values()))
#         intention_emb_seq = np.array(intention_emb_seq)
#         # 计算余弦相似度
#         similarity = cosine_similarity(intention_emb_seq, cluster_emb)
#         max_similarity_indices = np.argmax(similarity, axis=1)
#         intention_seq = max_similarity_indices.tolist()
#         return [intention_emb_seq, intention_seq]


class NNPM(nn.Module):
    def __init__(
        self,
        text_tokenizor,
        llm_model,
        prefix_length,
        region_num,
        hidden_dim,
        model_path,
        cluster_emb_path,
    ):
        super(NNPM, self).__init__()
        self.model_path = model_path
        self.prefix_model = Prefix_LLM(
            text_tokenizer=text_tokenizor,
            llm_model=llm_model,
            prefix_length=prefix_length,
            hidden_dim=hidden_dim,
        )
        self.prefix_model.load_state_dict(torch.load(model_path)["model_state_dict"])
        self.region_num = region_num
        self.region_proj = nn.Linear(hidden_dim, region_num)
        with open(cluster_emb_path, "r") as f:
            cluster_emb = json.load(f)
        cluster_emb = list(cluster_emb.values())
        cluster_emb.append(
            self.prefix_model.immob_ember.immob_embeddings.detach()
            .cpu()
            .numpy()
            .tolist()[0]
        )
        self.cluster_emb = torch.tensor(cluster_emb)
        print(1)

    def forward(
        self, disaster_level, tokens, masks, batch_size, normal_output, local_rank
    ):
        intention = self.prefix_model(
            disaster_level=disaster_level,
            tokens=tokens,
            masks=masks,
            batch_size=batch_size,
            local_rank=local_rank,
        )
        cosine_similarity_matrix = torch.matmul(
            intention, self.cluster_emb.cuda(local_rank).T
        )
        max_similarities, indices = cosine_similarity_matrix.max(dim=1)

        # 创建一个掩码，标记那些与最后一个向量最相似的样本
        mask = (
            indices != self.cluster_emb.shape[0]
        ).float()  # 如果索引不是15，则掩码值为1，否则为0

        output = self.region_proj(torch.mul(intention, normal_output))
        output = F.log_softmax(output, dim=1)
        return output, mask


class NNPMDataset(Dataset):
    def __init__(self, path, city, mode, model_name, normal_model):
        self.path = path
        self.trajectories = []
        self.embeddings_path = []
        self.model_name = model_name
        self.normal_model_path = normal_model + "Res"
        self.mode = mode
        select_mode = ["before", "dis"] if mode == "train" else ["after"]
        # select_mode = ["before"] if mode == "train" else ["after"]
        with open(f"{path}/util_datas/{mode}_dict.json", "r") as f:
            data_dict = json.load(f)
        for index, dname in enumerate(data_dict):
            mode_info = dname.split("-")[0]
            if (
                "_".join(mode_info.split("_")[:-1]) == city
                and mode_info.split("_")[-1] in select_mode
            ):
                self.trajectories.append(f"part_{index}.json")
                self.embeddings_path.append(f"{dname}.pt")
        # names = deque()
        # for dataset in datasets:
        #     for mode in modes:
        #         names.append(f"{dataset}-{mode}")
        # for index, f in enumerate(os.listdir(self.path)):
        #     if data_dict[index].split("-")[1] == "u_0_t_0":
        #         dataset_path, mode_path = names.popleft().split("-")
        #     if dataset_path == select_dataset and mode_path == "before":
        #         if os.path.isfile(os.path.join(self.path, f)):
        #             self.trajectories.append(f)
        #             self.embeddings_path.append(
        #                 f"/your_path/Mob_data/datas/{dataset_path}/dataset_tensor/{mode_path}/{data_dict[index]}.pt"
        #             )
        self.grid = len(gpd.read_file(f"{path}/{city}/grid.geojson"))

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj_path = self.trajectories[idx]
        with open(
            f"{self.path}/util_datas/traj_dataset/{self.model_name}/{self.mode}/{traj_path}",
            "r",
        ) as f:
            traj = json.load(f)
        NNPM_outputs = torch.load(
            f"{self.path}/util_datas/{self.normal_model_path}/{self.model_name}/{self.mode}/{self.embeddings_path[idx]}"
        )["out"]
        NNPM_outputs = torch.mean(NNPM_outputs, dim=0)
        max_len = 50
        ret_reg_seq = traj["reg_seq"][0:max_len]
        ret_reg_intention_emb_seq = traj["reg_intention_emb_seq"][0:max_len]
        ret_next_intention_emb = traj["next_intention_emb"]
        ret_next_reg = traj["next_reg"] - int(traj["next_reg"] / 1e8) * 1e8
        ret_disaster_level = [traj["disaster_level"]]
        ref_intention_emb_seq = [x[0:max_len] for x in traj["ref_intention_emb_seq"]]
        tokens = []
        masks = []
        for index, x in enumerate(traj["tokens"]):
            if len(x) == 1:
                tokens.append(x)
                masks.append(True)
                end_index = index + max_len
            else:
                if index < end_index:
                    tokens.append(x)
                    masks.append(False)
        tokens = [x[0] if len(x) == 1 else x for x in tokens]
        return (
            ret_reg_seq,
            ret_reg_intention_emb_seq,
            ret_next_intention_emb,
            ret_next_reg,
            ret_disaster_level,
            ref_intention_emb_seq,
            tokens,
            masks,
            NNPM_outputs,
        )

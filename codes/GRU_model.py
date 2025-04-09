import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from tqdm import *


class GRUModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        use_intention,
        strategy,
        rnn_model,
    ):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_model = rnn_model
        if rnn_model == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_model == "RNN":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_model == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.use_intention = use_intention
        self.strategy = strategy
        if not use_intention:
            self.fc = nn.Linear(hidden_size, output_size)
        else:
            if strategy == "mul":
                self.fc = nn.Linear(hidden_size, output_size)
            elif strategy == "concat" and use_intention:
                self.fc = nn.Linear(2 * hidden_size, output_size)
            elif strategy == "attn" and use_intention:
                self.attn = nn.Linear(2 * hidden_size, 1)  # 注意力层
                self.fc = nn.Linear(hidden_size, output_size)  # 最终的全连接层

    def forward(self, x, h0, intention=None):
        if self.rnn_model == "LSTM":
            x1, _ = self.rnn(x, (h0, h0))
        else:
            x1, _ = self.rnn(x, h0)
        x2 = x1[:, -1, :]
        if self.use_intention:
            if self.strategy == "mul":
                x2 = torch.mul(x2, intention)
            elif self.strategy == "concat":
                x2 = torch.concatenate([x2, intention], dim=1)
            elif self.strategy == "attn":
                seq_len = x1.size(1)  # 获取序列长度
                intention_expanded = intention.unsqueeze(1).expand(
                    -1, seq_len, -1
                )  # 扩展意图向量以匹配序列长度
                attn_input = torch.cat(
                    (x1, intention_expanded), dim=2
                )  # 拼接GRU输出和意图向量
                attn_weights = F.softmax(
                    self.attn(attn_input).squeeze(2), dim=1
                )  # 计算注意力权重
                x2 = torch.bmm(attn_weights.unsqueeze(1), x1).squeeze(1)  # 加权求和
        else:
            pass
        x3 = self.fc(x2)  # 只取最后一个时间步的输出
        x3 = F.log_softmax(x3, dim=1)
        return x3


class GRU_Dataset(Dataset):
    def __init__(self, mode, args, model_name):
        trajs = []
        city = args.city
        path = args.path
        self.use_intention = args.use_intention
        if mode == "train":
            stage = "before"
        elif args.scenario == "normal":
            stage = "before"
        else:
            stage = "after"
        set_mode = "train" if args.scenario == "normal" else mode
        with open(
            f"/your_path/Mob_data/datas/util_datas/{set_mode}_dict.json", "r"
        ) as f:
            data_dict = json.load(f)
        with open(f"{args.path}/util_datas/main_dataset/{city}/{stage}.json", "r") as f:
            data_source_1 = json.load(f)
        for index, x in tqdm(enumerate(data_dict), total=len(data_dict)):
            if city in x:
                if not x.replace(f"{city}_", "").split("-")[0] == stage:
                    print(1)
                    break
                    stage = x.replace(f"{city}_", "").split("-")[0]
                with open(
                    f"{path}/util_datas/NNPM_dataset/{model_name}/train/part_{index}.json",
                    "r",
                ) as f:
                    data_source_2 = json.load(f)
                trajs.append(
                    {
                        "st_emb_seq": data_source_1[0]["st_emb_seq"],
                        "next_grid": data_source_1[0]["grid_seq"][8],
                        "next_intention_emb": data_source_2["next_intention_emb"],
                        "next_intention": data_source_2["next_intention"],
                        "grid_seq": data_source_1[0]["grid_seq"],
                    }
                )
                data_source_1 = data_source_1[1:]
        with open(f"/your_path/Mob_data/datas/{city}/data_dis.pk", "rb") as f:
            data = pickle.load(f)
        self.region_num = len(data["vid_list"])
        if args.scenario == "normal":
            l = int(0.5 * len(trajs))
            self.trajs = trajs[:l] if mode == "train" else trajs[l:]
        else:
            self.trajs = trajs

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        traj = self.trajs[idx]
        if len(traj["st_emb_seq"]) <= 8:
            emb_seq = np.pad(
                np.array(traj["st_emb_seq"]),
                ((0, 8 - len(traj["st_emb_seq"])), (0, 0)),
                "constant",
            )
        else:
            emb_seq = np.array(traj["st_emb_seq"])[:8]
        return (
            torch.tensor(emb_seq, dtype=torch.float32),
            traj["next_grid"],
            torch.tensor(traj["next_intention_emb"], dtype=torch.float32),
            traj["grid_seq"][:9],
        )


def compute_mrr(logits, labels):
    """
    计算MRR（Mean Reciprocal Rank）

    :param logits: 二维数组，形状为 (num_samples, num_classes)，表示每个样本的预测分数
    :param labels: 一维数组，形状为 (num_samples,)，表示每个样本的真实标签
    :return: MRR值
    """
    num_samples = len(labels)
    reciprocal_ranks = []
    logits = logits.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    for i in range(num_samples):
        # 获取当前样本的预测分数
        sample_logits = logits[i]

        # 获取当前样本的真实标签
        true_label = labels[i]

        # 计算每个类别的排名
        ranks = np.argsort(-sample_logits)  # 按分数从高到低排序

        # 找到真实标签的排名
        rank_of_true_label = np.where(ranks == true_label)[0][0] + 1  # 排名从1开始

        # 计算倒数排名
        reciprocal_rank = 1 / rank_of_true_label
        reciprocal_ranks.append(reciprocal_rank)

    # 计算MRR
    mrr = np.mean(reciprocal_ranks)
    return mrr


# 计算 NDCG@10
def ndcg_at_k(predictions, labels, k=10):
    sorted_indices = torch.argsort(predictions, dim=1, descending=True)
    relevance = (sorted_indices == labels.unsqueeze(1)).float()

    gain = 1.0 / torch.log2(torch.arange(2, k + 2, device=predictions.device).float())
    dcg = (relevance[:, :k] * gain).sum(dim=1)

    ideal_sorted_relevance = torch.sort(relevance, dim=1, descending=True)[0]
    idcg = (ideal_sorted_relevance[:, :k] * gain).sum(dim=1)

    ndcg = (dcg / idcg).mean().item()
    return ndcg


def compute_immob_metrics(predictions, labels, immob_index):
    # 将预测值转换为类别索引
    predicted_labels = predictions.argmax(dim=1)
    correct_predictions = (predicted_labels == labels).float()
    immob_pred = torch.stack([1 - correct_predictions, correct_predictions], dim=1)
    # 将数据移到 CPU 并转换为 numpy 数组
    immob_pred = immob_pred.cpu().numpy()[:, 1]
    immob_label = np.zeros(len(labels), dtype=np.int64)
    immob_label[immob_index] = 1

    # 计算 Precision
    precision = precision_score(immob_label, immob_pred)

    # 计算 Recall
    recall = recall_score(immob_label, immob_pred)

    # 计算 F1 分数
    f1 = f1_score(immob_label, immob_pred)

    return precision, recall, f1


def cal_traj_metrics(input=None, output=None, label=None, init=True):
    if init == True:
        return {
            "Acc@1": [],
            "Acc@10": [],
            "Acc@100": [],
            "MRR": [],
            "NDCG@5": [],
            "NDCG@10": [],
            "Pre": [],
            "Rec": [],
            "F1": [],
        }
    else:
        _, pred_topk_1 = output.topk(1, dim=1, largest=True, sorted=True)
        _, pred_topk_10 = output.topk(10, dim=1, largest=True, sorted=True)
        _, pred_topk_100 = output.topk(100, dim=1, largest=True, sorted=True)
        # 计算 Acc@1
        correct_top1 = (pred_topk_1 == label.view(-1, 1)).any(dim=1).sum().item()
        acc_top1 = correct_top1 / len(label)

        # 计算 Acc@5
        correct_top10 = (pred_topk_10 == label.view(-1, 1)).any(dim=1).sum().item()
        acc_top10 = correct_top10 / len(label)

        # 计算 Acc@10
        correct_top100 = (pred_topk_100 == label.view(-1, 1)).any(dim=1).sum().item()
        acc_top100 = correct_top100 / len(label)
        mrr = compute_mrr(output, label)
        ndcg_5 = ndcg_at_k(output, label, k=5)
        ndcg_10 = ndcg_at_k(output, label, k=10)
        metrics = {
            "Acc@1": acc_top1,
            "Acc@10": acc_top10,
            "Acc@100": acc_top100,
            "MRR": mrr,
            "NDCG@5": ndcg_5,
            "NDCG@10": ndcg_10,
        }
        # Immob
        # decide which is immob
        immob_index = np.where((input[:, -1] - label).detach().cpu().numpy() == 0)[0]
        if not len(immob_index) == 0:
            precision, recall, f1 = compute_immob_metrics(
                output, input[:, -1], immob_index
            )
            metrics["Pre"] = precision
            metrics["Rec"] = recall
            metrics["F1"] = f1
        else:
            metrics["Pre"] = None
            metrics["Rec"] = None
            metrics["F1"] = None

        return metrics


def tb_metric_weight(writer, train_plot_data, test_plot_data, epoch):
    # 记录训练损失和指标
    [train_metrics, train_loss] = train_plot_data
    [test_metrics, test_loss] = test_plot_data
    writer.add_scalars("Loss", {"train": train_loss, "test": test_loss}, epoch)
    for k in train_metrics.keys():
        writer.add_scalars(
            f"Metrics/{k}",
            {"train": np.mean(train_metrics[k]), "test": np.mean(test_metrics[k])},
            epoch,
        )

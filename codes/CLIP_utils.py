import torch
import torch.nn as nn
from tqdm import *
from sklearn.metrics import mean_squared_error
import numpy as np
import torch.nn.functional as F


def train_model(
    model,
    dataloader,
    optimizer,
    criterion,
    mode,
    cluster_emb,
    immob=None,
):
    if immob == None:
        immob = dataloader.dataset.immob
    if mode == "train":
        model.train()
    # else:
    #     model.eval()
    total_loss = []
    info_nce_losses = []
    cross_entropy_losses = []
    total_metrics = cal_CLIP_metrics(init=True)
    cluster_intentions = list(cluster_emb.values())
    cluster_intentions.append(model.immob_emb.detach().cpu().numpy().tolist()[0])
    cluster_intentions = torch.tensor(cluster_intentions, dtype=torch.float32).cuda()

    for trajectories, intentions, intention_index in tqdm(dataloader, desc=f"{mode}"):
        trajectories = (
            torch.tensor(trajectories, dtype=torch.float32).cuda().unsqueeze(1)
        )
        intentions = torch.tensor(intentions, dtype=torch.float32).cuda()
        intention_index = intention_index.cuda()
        if mode == "train":
            optimizer.zero_grad()
        trajectory_embedding, intention_embedding, intention_pred = model(
            trajectories, intentions
        )
        _, cluster_intention_embedding, _ = model(
            trajectories[: cluster_intentions.shape[0], :], cluster_intentions
        )
        loss, info_nce_loss, cross_entropy_loss = criterion(
            trajectory_embedding, intention_embedding, intention_pred, intention_index
        )
        total_loss.append(loss.item())
        info_nce_losses.append(info_nce_loss.item())
        cross_entropy_losses.append(cross_entropy_loss.item())
        if mode == "train":
            loss.backward()
            optimizer.step()
        metrics = cal_CLIP_metrics(
            trajectory_embedding=trajectory_embedding,
            intention_embedding=intention_embedding,
            cluster_intention_embedding=cluster_intention_embedding,
            intention_index=intention_index,
            init=False,
            immob=immob,
        )
        for k, v in metrics.items():
            total_metrics[k].append(v)

    total_loss = np.mean(total_loss)
    info_nce_loss = np.mean(info_nce_losses)
    cross_entropy_loss = np.mean(cross_entropy_losses)

    for k, v in metrics.items():
        total_metrics[k] = np.mean(v)

    return total_loss, info_nce_loss, cross_entropy_loss, model, total_metrics


def cal_CLIP_metrics(
    trajectory_embedding=None,
    intention_embedding=None,
    cluster_intention_embedding=None,
    intention_index=None,
    init=True,
    immob=None,
):
    if init == True:
        return {"Acc@1": [], "Acc@5": [], "Acc@10": [], "RMSE": []}
    else:
        # 使用cosine_similarity函数计算相似度
        similarities = nn.functional.cosine_similarity(
            trajectory_embedding.unsqueeze(1),
            cluster_intention_embedding.unsqueeze(0),
            dim=2,
        )
        # 找到每个A的行对应的最大相似度及其索引
        best_similarities, indices = torch.max(similarities, dim=1)
        _, pred_topk_1 = similarities.topk(1, dim=1, largest=True, sorted=True)
        _, pred_topk_5 = similarities.topk(5, dim=1, largest=True, sorted=True)
        _, pred_topk_10 = similarities.topk(10, dim=1, largest=True, sorted=True)
        # 计算 Acc@1
        correct_top1 = (
            (pred_topk_1 == intention_index.view(-1, 1)).any(dim=1).sum().item()
        )
        acc_top1 = correct_top1 / len(intention_index)

        # 计算 Acc@5
        correct_top5 = (
            (pred_topk_5 == intention_index.view(-1, 1)).any(dim=1).sum().item()
        )
        acc_top5 = correct_top5 / len(intention_index)

        # 计算 Acc@10
        correct_top10 = (
            (pred_topk_10 == intention_index.view(-1, 1)).any(dim=1).sum().item()
        )
        acc_top10 = correct_top10 / len(intention_index)
        rmse = mean_squared_error(
            F.normalize(trajectory_embedding, dim=-1).detach().cpu().numpy(),
            F.normalize(cluster_intention_embedding, dim=-1)
            .detach()
            .cpu()
            .numpy()[intention_index.detach().cpu().numpy().tolist()],
        )
        metrics = {
            "Acc@1": acc_top1,
            "Acc@5": acc_top5,
            "Acc@10": acc_top10,
            "RMSE": rmse,
        }
        return metrics


def tb_metric_weight(writer, train_plot_data, test_plot_data, epoch):
    # 记录训练损失和指标
    [
        train_metrics,
        train_loss,
        train_info_nce_loss,
        train_cross_entropy_loss,
    ] = train_plot_data
    [
        test_metrics,
        test_loss,
        test_info_nce_loss,
        test_cross_entropy_loss,
    ] = test_plot_data
    writer.add_scalars("Loss", {"train": train_loss, "test": test_loss}, epoch)
    writer.add_scalars(
        "InfoNCELoss", {"train": train_info_nce_loss, "test": test_info_nce_loss}, epoch
    )
    writer.add_scalars(
        "CrossEntropyLoss",
        {"train": train_cross_entropy_loss, "test": test_cross_entropy_loss},
        epoch,
    )
    for k in train_metrics.keys():
        writer.add_scalars(
            f"Metrics/{k}",
            {"train": np.mean(train_metrics[k]), "test": np.mean(test_metrics[k])},
            epoch,
        )

import pandas as pd
import json
import numpy as np
import networkx as nx
from tqdm import *
import ast
import logging
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import gc
import re
import time
import ast
import os
import gc


def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def get_disaster_data(dataset):
    with open(
        f"/data3/tangyinzhou/Mob_data/datas/{dataset}/rain_data/region_name_c2e.json",
        "r",
    ) as f:
        c2e_region = json.load(f)
    regions = list(c2e_region.values())
    sum_raindata = []
    for reg_index, reg in enumerate(regions):
        region_rain_data = pd.read_csv(
            f"/home/tangyinzhou/PostDisasterMobPred/Data/datas/{dataset}/rain_data/raw_rain_data/{reg}_Daily.csv"
        )
        x = region_rain_data["date"].tolist()
        y = region_rain_data["precipitation"].tolist()
        sum_raindata.append(y)
    dates = x.copy()
    rain = np.sum(np.array(sum_raindata), axis=0) / (reg_index + 1)
    return dates, rain


def get_disaster_level(rain_level):
    if rain_level <= 1:
        dis_level = "no disaster"
    elif 1 < rain_level <= 25:
        dis_level = "minor disaster"
    elif 25 < rain_level <= 50:
        dis_level = "general disaster"
    elif rain_level > 50:
        dis_level = "severe disaster"
    return dis_level


def reconstruct_multigraph_from_txt(dataset):
    # 加载节点s
    with open(f"/data3/tangyinzhou/Mob_data/datas/{dataset}/nodes.txt", "r") as f:
        nodes = [float(line.strip()) for line in f.readlines()]

    # 加载节点属性
    node_attributes = {}
    with open(
        f"/data3/tangyinzhou/Mob_data/datas/{dataset}/node_attributes.txt", "r"
    ) as f:
        for line in f:
            parts = line.strip().split("|")
            node = float(parts[0])
            attrs = {}
            for part in parts[1:]:
                key, value = part.split("=")
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                attrs[key] = value
            node_attributes[node] = attrs

    # 加载边
    edges = []
    with open(f"/data3/tangyinzhou/Mob_data/datas/{dataset}/edges.txt", "r") as f:
        for line in tqdm(f, desc="loading edges"):
            u, v, k = map(float, line.strip().split("|"))
            edges.append((u, v, k))

    # 加载边属性
    edge_attributes = {}
    with open(
        f"/data3/tangyinzhou/Mob_data/datas/{dataset}/edge_attributes.txt", "r"
    ) as f:
        for line in tqdm(f, desc="loading edge attrs"):
            parts = line.strip().split("|")
            u, v, k = map(float, parts[:3])
            attrs = {}
            etype = parts[3].split("=")[1]
            if etype == "road":
                for part in parts[3:]:
                    key, value = part.split("=")
                    attrs[key] = value
            elif etype in ["shortest_path", "connected_paths"]:
                for part in parts[3:]:
                    key, value = part.split("=")
                    if key == "type":
                        attrs[key] = value
                    else:
                        attrs[key] = int(value.split("\n")[0])
            elif etype == "road_cat":
                for part in parts[3:]:
                    key, value = part.split("=")
                    if key == "type":
                        attrs[key] = value
                    else:
                        attrs[key] = ast.literal_eval(value.split("\n")[0])
            edge_attributes[(u, v, k)] = attrs

    # 重构多图
    G = nx.MultiGraph()

    # 添加节点和节点属性
    for node in tqdm(nodes, desc="adding nodes"):
        G.add_node(node, **node_attributes.get(node, {}))

    # 添加边和边属性
    for u, v, k in tqdm(edges, desc="adding edges"):
        G.add_edge(u, v, key=k, **edge_attributes.get((u, v, k), {}))

    return G


def get_modelname_and_hiddim(use_LLM: str):
    if use_LLM == "meta-llama/Meta-Llama-3-8B":
        hidden_dim = 4096  # 隐藏层维度，与BERT一致
        model_name = "Meta-Llama-3-8B"
    elif use_LLM == "meta-llama/Llama-3.2-1B":
        hidden_dim = 2048  # 隐藏层维度，与BERT一致
        model_name = "Meta-Llama-3.2-1B"
    elif use_LLM == "meta-llama/Llama-3.2-1B-Instruct":
        hidden_dim = 2048  # 隐藏层维度，与BERT一致
        model_name = "Meta-Llama-3.2-1B"
    elif use_LLM == "meta-llama/Meta-Llama-3-8B-Instruct":
        hidden_dim = 4096  # 隐藏层维度，与BERT一致
        model_name = "Meta-Llama-3-8B"
    else:
        print("no given LLM")
        exit(0)
    return model_name, hidden_dim


def get_train_test_set(target_city: str):
    datasets = [
        "qingyuan",
        "shaoguan",
        "hezhou",
        "zhongshan_01",
        "zhuhai_01",
        "wuzhou_01",
        "handan",
    ]
    modes = ["before", "dis", "after"]
    train = []
    test = []
    for city in datasets:
        if not city == target_city:
            for mode in modes:
                train.append(f"{city}-{mode}")
        else:
            train.append(f"{city}-before")
            train.append(f"{city}-dis")
            test.append(f"{city}-after")
    return train, test


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def cal_traj_metrics(output=None, label=None, init=True):
    if init == True:
        return {"Acc@1": [], "Acc@5": [], "Acc@10": []}
    else:
        _, pred_topk_1 = output.topk(1, dim=1, largest=True, sorted=True)
        _, pred_topk_5 = output.topk(5, dim=1, largest=True, sorted=True)
        _, pred_topk_10 = output.topk(10, dim=1, largest=True, sorted=True)
        # 计算 Acc@1
        correct_top1 = (pred_topk_1 == label.view(-1, 1)).any(dim=1).sum().item()
        acc_top1 = correct_top1 / len(label)

        # 计算 Acc@5
        correct_top5 = (pred_topk_5 == label.view(-1, 1)).any(dim=1).sum().item()
        acc_top5 = correct_top5 / len(label)

        # 计算 Acc@10
        correct_top10 = (pred_topk_10 == label.view(-1, 1)).any(dim=1).sum().item()
        acc_top10 = correct_top10 / len(label)
        metrics = {"Acc@1": acc_top1, "Acc@5": acc_top5, "Acc@10": acc_top10}
        return metrics


def find_and_trim(s, substrings):
    # 初始化最小索引为无穷大
    min_index = float("inf")
    chosen_substring = None

    # 遍历列表中的每个子字符串
    for substring in substrings:
        index = s.find(substring)
        if index != -1 and index < min_index:
            min_index = index
            chosen_substring = substring

    # 如果没有找到任何一个子字符串，返回原字符串
    if min_index == float("inf"):
        return s, chosen_substring
    s = s[min_index + 3 :]
    # 返回从找到的位置开始的子串
    return s, chosen_substring


def get_pred_label(row):
    row = row.lower()
    row, label1 = find_and_trim(row, ["yes", "no"])
    if not label1 == None:
        row, label2 = find_and_trim(row, ["yes", "no"])
        if not label2 == None:
            row, label3 = find_and_trim(
                row, ["yes", "no"] + [str(x) for x in list(range(16))]
            )
            if not label3 == None:
                return [label1, label2, label3]
    return None


def get_ground_truth(s):
    s = s.strip("[]")

    # 使用逗号分隔字符串
    elements = s.split(",")

    # 去除每个元素的首尾空白字符
    elements = [element.strip() for element in elements]

    return elements


def cal_intention_metrics(
    output=None, label=None, init=True, mode=None, tokenizer=None
):
    if init == True:
        return {
            "Format Tokens": [],
            "Acc@Change": [],
            "Acc@Immob": [],
            "Acc@Intention": [],
        }
    else:
        # cal format tokens
        if mode == "train":
            output = output.detach().cpu().numpy()[:, -7:]
            label = label.detach().cpu().numpy()[:, -7:]
        else:
            output = output.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

        batch_size = output.shape[0]
        ft_acc = []
        pred = tokenizer.batch_decode(output, skip_special_tokens=True)
        ground_truth = tokenizer.batch_decode(label, skip_special_tokens=True)
        pattern = r"^\[\s*(\w+\s*,\s*)*\w+\s*\]$"
        ft_acc = [0 if re.match(pattern, row) == None else 1 for row in pred]
        ft_acc = np.mean(ft_acc)
        # Acc@Change
        change_acc = []
        # Acc@Immob
        immob_acc = []
        # Acc@Intention
        intention_acc = []
        # for index in range(len(pred)):
        #     if re.match(pattern, pred[index]):
        #         list_pattern = r"\[(.*?)\]"
        #         match = re.search(list_pattern, pred[index])
        #         if match:
        #             content = match.group(1)  # 提取方括号内的内容
        #             # 分割内容为列表
        #             batch_pred = [item.strip() for item in content.split(",")]
        #             if len(batch_pred) == 3:
        #                 batch_label = [
        #                     item.strip()
        #                     for item in re.search(list_pattern, ground_truth[index])
        #                     .group(1)
        #                     .split(",")
        #                 ]
        #                 change_acc.append(1 if batch_pred[0] == batch_label[0] else 0)
        #                 immob_acc.append(1 if batch_pred[1] == batch_label[1] else 0)
        #                 intention_acc.append(
        #                     1 if batch_pred[2] == batch_label[2] else 0
        #                 )
        #                 continue
        #     change_acc.append(0)
        #     immob_acc.append(0)
        #     intention_acc.append(0)
        for index in range(len(pred)):
            sample_pred = pred[index]
            sample_ground_truth = ground_truth[index]
            sample_pred = get_pred_label(sample_pred)
            sample_ground_truth = get_ground_truth(ground_truth[index])
            if sample_pred == None:
                change_acc.append(0)
                immob_acc.append(0)
                intention_acc.append(0)
            else:
                if sample_ground_truth[0] == "yes":
                    change_acc.append(1 if sample_pred[0] == "yes" else 0)
                    immob_acc.append(1 if sample_pred[0] == "yes" else 0)
                    intention_acc.append(1 if sample_pred[0] == "yes" else 0)
                else:
                    change_acc.append(1 if sample_pred[0] == "no" else 0)
                    if sample_ground_truth[1] == "yes":
                        immob_acc.append(1 if sample_pred[1] == "yes" else 0)
                        intention_acc.append(1 if sample_pred[1] == "yes" else 0)
                    else:
                        immob_acc.append(1 if sample_pred[1] == "no" else 0)
                        intention_acc.append(
                            1 if sample_pred[2] == sample_pred[2] else 0
                        )
        return {
            "Format Tokens": ft_acc,
            "Acc@Change": np.mean(change_acc),
            "Acc@Immob": np.mean(immob_acc),
            "Acc@Intention": np.mean(intention_acc),
        }


def train_test_NNPM_model(
    args, pred_model, dataloader, dataset, mode, loss_fn, optimizer
):
    if mode == "train":
        pred_model.train()
    else:
        pred_model.eval()
    metrics_list = cal_traj_metrics(init=True)
    total_loss = []
    for index, (
        reg_seq,
        reg_intention_emb_seq,
        next_intention_emb,
        next_reg,
        disaster_level,
        ref_intention_emb_seq,
        tokens,
        masks,
        normal_output,
    ) in tqdm(
        enumerate(dataloader),
        total=int(dataset.__len__() / args.batch_size),
    ):
        tokens = [torch.stack(x, dim=1).cuda(args.local_rank) for x in tokens]
        masks = [x.cuda(args.local_rank) for x in masks]
        output, mask = pred_model(
            disaster_level=disaster_level[0],
            tokens=tokens,
            masks=masks,
            batch_size=args.batch_size,
            local_rank=args.local_rank,
            normal_output=normal_output.cuda(args.local_rank),
        )
        output = output * mask.unsqueeze(-1)
        labels = (next_reg.cuda(args.local_rank) * mask).to(torch.int64)
        loss = loss_fn(output, labels)
        total_loss.append(loss.item())
        if mode == "train":
            loss.backward()
        metrics = cal_traj_metrics(output=output, label=labels, init=False)
        for k, v in metrics.items():
            metrics_list[k].append(v)
        if index % 5 == 0 and mode == "train":
            print(f"Loss for device {args.local_rank}:{loss.item()}")
            pmetrics = [f"{k}:{np.mean(v)}" for k, v in metrics_list.items()]
            print("||".join(pmetrics))
            optimizer.step()
            optimizer.zero_grad()
        gc.collect()
    if (index + 1) % 5 != 0 and mode == "train":
        optimizer.step()
        optimizer.zero_grad()
    return np.mean(total_loss), metrics_list, pred_model, optimizer


def train_test_LLM_model(
    args,
    prefix_model,
    dataloader,
    dataset,
    mode,
    loss_fn,
    optimizer,
    model_name,
    logger,
):
    metrics_list = cal_intention_metrics(init=True)
    total_loss = []
    for index, (
        next_intention,
        disaster_level,
        tokens,
        masks,
        label,
        label_marker,
        pred_label,
    ) in tqdm(
        enumerate(dataloader),
        total=(
            int(dataset.__len__() / (args.batch_size * args.gpu_num))
            if args.multi_gpu
            else int(dataset.__len__() / (args.batch_size))
        ),
        desc=f"{mode}ing for device {args.local_rank}",
    ):
        tokens = [torch.stack(x, dim=1).cuda(args.local_rank) for x in tokens]
        masks = [x.cuda(args.local_rank) for x in masks]
        if not args.generate_answer:
            if mode == "train":
                prefix_model.train()
            else:
                prefix_model.eval()
            output, logits = prefix_model(
                disaster_level=disaster_level[0],
                tokens=tokens,
                masks=masks,
                batch_size=args.batch_size,
                local_rank=args.local_rank,
            )
            label = torch.stack(label).T
            label_marker = torch.stack(label_marker)[:, 0]
            logits = logits[:, label_marker, :].transpose(1, 2)
            label = label[:, label_marker].to(logits.device)
            loss = loss_fn(logits, label)
            total_loss.append(loss.item())
            if mode == "train":
                loss.backward()
                if index % 10 == 0 and args.local_rank == 0:
                    text_output = (
                        prefix_model.module.generate_text(
                            disaster_level=disaster_level[0],
                            tokens=tokens[:-1],
                            masks=masks[:-1],
                            local_rank=args.local_rank,
                        )
                        if args.multi_gpu
                        else prefix_model.generate_text(
                            disaster_level=disaster_level[0],
                            tokens=tokens[:-1],
                            masks=masks[:-1],
                            local_rank=args.local_rank,
                        )
                    )
                    text = (
                        prefix_model.module.text_tokenizer.batch_decode(text_output)
                        if args.multi_gpu
                        else prefix_model.text_tokenizer.batch_decode(text_output)
                    )
                    for tid, t in enumerate(text):
                        logger.info(f"{mode} sample {tid}: {t}")
            else:
                text_output = (
                    prefix_model.module.generate_text(
                        disaster_level=disaster_level[0],
                        tokens=tokens,
                        masks=masks,
                        local_rank=args.local_rank,
                    )
                    if args.multi_gpu
                    else prefix_model.generate_text(
                        disaster_level=disaster_level[0],
                        tokens=tokens[:-1],
                        masks=masks[:-1],
                        local_rank=args.local_rank,
                    )
                )
                label = torch.stack(pred_label).T.to(output.device)

                text = (
                    prefix_model.module.text_tokenizer.batch_decode(text_output)
                    if args.multi_gpu
                    else prefix_model.text_tokenizer.batch_decode(text_output)
                )
                for tid, t in enumerate(text):
                    logger.info(f"{mode} sample {tid}: {t}")
        else:
            text_output = (
                prefix_model.module.generate_text(
                    disaster_level=disaster_level[0],
                    tokens=tokens[:-1] if mode == "train" else tokens,
                    masks=masks[:-1] if mode == "train" else masks,
                    local_rank=args.local_rank,
                )
                if args.multi_gpu
                else prefix_model.generate_text(
                    disaster_level=disaster_level[0],
                    tokens=tokens[:-1] if mode == "train" else tokens,
                    masks=masks[:-1] if mode == "train" else masks,
                    local_rank=args.local_rank,
                )
            )

            text = (
                prefix_model.module.text_tokenizer.batch_decode(text_output)
                if args.multi_gpu
                else prefix_model.text_tokenizer.batch_decode(text_output)
            )
            # save vars
            for x, t in enumerate(text):
                with open(
                    f"{args.path}/util_datas/traj_dataset/{model_name}/{mode}/part_{index*len(text)+x}.json"
                ) as f:
                    sample_data = json.load(f)
                    t = t.strip("[]").split(",")
                    if t[0] == "yes":
                        pred_intention = sample_data["next_intention"]
                    elif t[1] == "yes":
                        pred_intention = 15
                    else:
                        pred_intention = int(t[2])
                    sample_data["pred_intention"] = pred_intention
                os.makedirs(
                    f"{args.path}/util_datas/NNPM_dataset/{model_name}/{mode}",
                    exist_ok=True,
                )
                with open(
                    f"{args.path}/util_datas/NNPM_dataset/{model_name}/{mode}/part_{index*len(text)+x}.json",
                    "w",
                ) as f:
                    json.dump(sample_data, f)
        if not args.generate_answer:
            metrics = cal_intention_metrics(
                output=output if mode == "train" else text_output,
                label=label,
                init=False,
                mode=mode,
                tokenizer=(
                    prefix_model.module.text_tokenizer
                    if args.multi_gpu
                    else prefix_model.text_tokenizer
                ),
            )
            for k, v in metrics.items():
                metrics_list[k].append(v)
            if index % 5 == 0:
                optimizer.step()
                optimizer.zero_grad()
            # 删除变量
            # del tokens
            # del masks
            # del label
            # del label_marker
            # del output

            # 强制垃圾回收
            # gc.collect()

            # 清空缓存
            # torch.cuda.empty_cache()
    if not args.generate_answer:
        if (index + 1) % 5 != 0:
            optimizer.step()
            optimizer.zero_grad()
        for k, v in metrics_list.items():
            metrics_list[k] = np.mean(v)
        return np.mean(total_loss), metrics_list, prefix_model, optimizer


def tb_metric_weight(writer, train_plot_data, test_plot_data, epoch):
    # 记录训练损失和指标
    [train_metrics, train_loss] = train_plot_data
    [test_metrics, test_loss] = test_plot_data
    writer.add_scalars("Loss", {"train": train_loss, "test": test_loss}, epoch)
    for k in train_metrics.keys():
        writer.add_scalars(
            f"Metrics/{k}", {"train": train_metrics[k], "test": test_metrics[k]}, epoch
        )


# 配置 logging
def setup_logging(log_file):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)  # 设置日志级别为 DEBUG

    # 创建一个 handler 用于写入日志文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # 创建一个 handler 用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 创建一个 formatter 来设置日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 将 formatter 添加到 handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将 handlers 添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def print_memory_usage(logger):
    memory_dict = defaultdict(int)

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                tensor = obj
                mem = tensor.element_size() * tensor.nelement()
                memory_dict[str(tensor.size())] += mem
        except:
            pass  # ignore if error occurs during checking

    # Sort by memory usage
    sorted_memory = sorted(memory_dict.items(), key=lambda item: item[1], reverse=True)

    for size, mem in sorted_memory:
        logger.debug(
            f"Tensor of size {size} is using {mem / (1024**2):.2f} MB of GPU memory."
        )


def get_formats(tokenizer):
    formats = []
    for o1 in ["yes", "no"]:
        for o2 in ["yes", "no"]:
            for intention in range(16):
                formats.append(f"[{o1},{o2},{intention}]")
    fs = []
    for f in formats:
        f = tokenizer(str(f))["input_ids"][1:]
        fs.append(f)
    max_length = max(len(sublist) for sublist in fs)
    fs = [
        sublist + [tokenizer.eos_token_id] * (max_length - len(sublist))
        for sublist in fs
    ]
    fs = np.array(fs)
    unique_columns = [np.unique(fs[:, i]) for i in range(fs.shape[1])]
    return unique_columns


# def save_model(
#     epoch, prefix_model, args, optimizer, loss, metrics, logger, mode="loss"
# ):
#     torch.save(
#         {
#             "epoch": epoch + 1,
#             "model_state_dict": (
#                 prefix_model.module.prefix.state_dict()
#                 if args.multi_gpu
#                 else prefix_model.prefix.state_dict()
#             ),
#             "optimizer_state_dict": optimizer.state_dict(),
#             "loss": loss,
#             "Acc@Intention": metrics["Acc@Intention"],
#             "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         },
#         f"{save_path}/{model_name}_train_min_loss.pth",
#     )
#     min_loss = loss
#     logger.info(f"Saved train model at epoch {epoch} for min loss")
#     if mode == "loss":
#         return min_loss
#     else:
#         return metrics["Acc@Intention"]

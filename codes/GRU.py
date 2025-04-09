from GRU_model import *
import json
import argparse
import torch.optim as optim
import geopandas as gpd
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=500)
parser.add_argument("--city", type=str, default="zhongshan_01")
parser.add_argument("--target_city", type=str, default="zhongshan_01")
parser.add_argument("--input_dim", type=int, default=111)
parser.add_argument("--hidden_dim", type=int, default=4096)
parser.add_argument("--num_layers", type=int, default=8)
parser.add_argument("--path", type=str, default="/your_path/Mob_data/datas")
parser.add_argument("--use_LLM", type=str, default="meta-llama/Llama-3.2-8B")
parser.add_argument("--use_intention", type=str2bool, default=False)
parser.add_argument("--strategy", type=str, default="attn")
parser.add_argument("--scenario", type=str, default="disaster")
parser.add_argument("--model", type=str, default="LSTM")
args = parser.parse_args()


def get_modelname_and_hiddim(use_LLM: str):
    if use_LLM == "meta-llama/Meta-Llama-3-8B":
        hidden_dim = 4096  # 隐藏层维度，与BERT一致
        model_name = "Meta-Llama-3-8B"
    elif use_LLM == "meta-llama/Llama-3.2-1B":
        hidden_dim = 2048  # 隐藏层维度，与BERT一致
        model_name = "Meta-Llama-3.2-1B"
    else:
        print("no given LLM")
        exit(0)
    return model_name, hidden_dim


if __name__ == "__main__":
    model_name, hidden_dim = get_modelname_and_hiddim(args.use_LLM)
    os.makedirs(
        f"/your_path/Mob_data/datas/util_datas/GRURes/{model_name}",
        exist_ok=True,
    )
    # init dataset
    train_dataset = GRU_Dataset(mode="train", args=args, model_name=model_name)
    test_dataset = GRU_Dataset(mode="test", args=args, model_name=model_name)
    print("init dataset")
    # set save path
    save_path = f"{args.path}/util_datas/GRURes/{model_name}"
    os.makedirs(save_path, exist_ok=True)

    grids = gpd.read_file(f"{args.path}/{args.city}/grid.geojson")
    region_num = len(grids)
    model = GRUModel(
        input_size=args.input_dim,
        hidden_size=hidden_dim,
        num_layers=args.num_layers,
        output_size=region_num,
        use_intention=args.use_intention,
        strategy=args.strategy,
        rnn_model=args.model,
    ).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-6)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=True
    )
    min_train_loss = 1e8
    min_test_loss = 1e8
    max_train_acc = 0
    max_test_acc = 0
    best_metrics = cal_traj_metrics(init=True)
    for k, v in best_metrics.items():
        best_metrics[k] = 0
    for epoch in range(args.num_epochs):
        model.train()
        # epoch train
        train_loss = []
        train_metrics = cal_traj_metrics(init=True)
        for seq, label, intention, grid_seq in train_dataloader:
            seq = seq.cuda()
            label = label.cuda()
            intention = intention.cuda()
            grid_seq = torch.stack(grid_seq).T.cuda()
            h0 = torch.zeros(args.num_layers, seq.size(0), hidden_dim).cuda()
            pred_label = model(seq, h0, intention if args.use_intention else None)
            loss = criterion(pred_label, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            for k, v in cal_traj_metrics(
                input=grid_seq, output=pred_label, label=label, init=False
            ).items():
                if v:
                    train_metrics[k].append(v)
        train_loss = np.mean(train_loss)
        for k, v in train_metrics.items():
            train_metrics[k] = np.mean(v)

        print(
            f"Epoch: {epoch}, Train Loss: {loss}, Acc@1: {train_metrics['Acc@1']}, Acc@10: {train_metrics['Acc@10']}, MRR: {train_metrics['MRR']}, NDCG@5:{train_metrics['NDCG@5']}, NDCG@10:{train_metrics['NDCG@10']}, Pre:{train_metrics['Pre']}, Rec:{train_metrics['Rec']}, F1:{train_metrics['F1']}"
        )
        # save latest
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "loss": train_loss,
                "Acc@1": train_metrics["Acc@1"],
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            f"{save_path}/{args.city}_latest.pt",
        )
        # epoch test
        test_loss = []
        test_metrics = cal_traj_metrics(init=True)
        # model.eval()
        for seq, label, intention, grid_seq in test_dataloader:
            seq = seq.cuda()
            label = label.cuda()
            intention = intention.cuda()
            grid_seq = torch.stack(grid_seq).T.cuda()
            h0 = torch.zeros(args.num_layers, seq.size(0), hidden_dim).cuda()
            pred_label = model(seq, h0, intention if args.use_intention else None)
            loss = criterion(pred_label, label)
            test_loss.append(loss.item())
            for k, v in cal_traj_metrics(
                input=grid_seq, output=pred_label, label=label, init=False
            ).items():
                if v:
                    test_metrics[k].append(v)
        test_loss = np.mean(test_loss)
        for k, v in test_metrics.items():
            test_metrics[k] = np.mean(v)
        for k, v in best_metrics.items():
            if test_metrics[k] >= best_metrics[k]:
                best_metrics[k] = test_metrics[k]

        max_test_acc = max(max_test_acc, test_metrics["Acc@1"])
        print(
            f"Epoch: {epoch}, Test Loss: {test_loss}, Acc@1: {best_metrics['Acc@1']}, Acc@10: {best_metrics['Acc@10']}, MRR: {best_metrics['MRR']}, NDCG@5:{best_metrics['NDCG@5']}, NDCG@10:{best_metrics['NDCG@10']}, Pre:{test_metrics['Pre']}, Rec:{test_metrics['Rec']}, F1:{best_metrics['F1']}"
        )
        print("=" * 100)

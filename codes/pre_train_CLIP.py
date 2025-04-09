from CLIP_model import *
import warnings
from torch import optim
import argparse
from datetime import datetime
import os
from CLIP_utils import *
from torch.utils.tensorboard import SummaryWriter  # Create an instance of the object
from torch.utils.data import ConcatDataset

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--input_dim", type=int, default=111)
parser.add_argument("--intention_dim", type=int, default=20)
parser.add_argument("--traj_pad_length", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--attn_dim", type=int, default=512)
parser.add_argument("--proto_dim", type=int, default=1024)
parser.add_argument("--use_pretrained", type=str2bool, default=False)
parser.add_argument("--use_LLM", type=str, default="meta-llama/Llama-3.2-8B")
parser.add_argument("--target_city", type=str, default="zhongshan_01")
parser.add_argument("--path", type=str, default="/your_path/Mob_data/datas")
parser.add_argument("--LLM_path", type=str, default="/your_path/LLMs")
args = parser.parse_args()

# 主程序
if __name__ == "__main__":
    train, test = get_train_test_set(target_city=args.target_city)
    model_name, hidden_dim = get_modelname_and_hiddim(args.use_LLM)
    log_dir = f"{args.path}/util_datas/tensorboard_logs/{model_name}/CLIP"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    model = AutoModel.from_pretrained(args.use_LLM, cache_dir=args.LLM_path)
    save_path = f"{args.path}/util_datas/CLIP_checkpoints/{model_name}"
    os.makedirs(save_path, exist_ok=True)
    text_vocab = model.embed_tokens.weight
    # init model
    cluster_model = Cluster(
        datasets=train, max_cluster_num=50, have_c=True, path=args.path
    )
    print("built cluster")
    clip_model = CLIPModel(
        dataset=train,
        path=args.path,
        traj_pad_length=args.traj_pad_length,
        input_dim=args.input_dim,
        hidden_dim=hidden_dim,
        proto_dim=args.proto_dim,
        attn_dim=args.attn_dim,
        intention_dim=args.intention_dim,
        text_vocab=text_vocab,
        use_LLM=args.use_LLM,
        llm_model=model,
        class_num=16,
    ).cuda()
    # build dataset
    train_dataset = CLIPDataset(
        datasets=train,
        traj_pad_length=args.traj_pad_length,
        cluster=cluster_model,
        path=args.path,
    )
    test_dataset = CLIPDataset(
        datasets=test,
        traj_pad_length=args.traj_pad_length,
        cluster=cluster_model,
        path=args.path,
        city_immob_indexs=train_dataset.city_immob_indexs,
    )
    print("built dataset")

    # build dataloader
    train_dataloader = DataLoader(
        ConcatDataset([train_dataset, test_dataset]),
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = optim.AdamW(clip_model.parameters(), lr=5e-5)
    criterion = HybridLoss(alpha=0.5, beta=0.5)

    # train model
    max_train_acc = 0
    max_test_acc = 0
    min_train_loss = 1e8
    min_test_loss = 1e8
    print("start training")

    # load checkpoints
    checkpoint_path = f"{save_path}/{model_name}_latest.pth"
    if os.path.exists(checkpoint_path) and args.use_pretrained == True:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        train_loss = checkpoint["loss"]
        print(f"Loaded checkpoint from {start_epoch}")
    else:
        start_epoch = 0

    # start train and eval
    for epoch in range(start_epoch, args.num_epochs):
        # train
        (
            train_loss,
            train_info_nce_loss,
            train_cross_entropy_loss,
            trained_model,
            train_metrics,
        ) = train_model(
            model=clip_model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            mode="train",
            cluster_emb=cluster_model.cluster_emb,
            immob=test_dataloader.dataset.immob,
        )
        print(f"Epoch {epoch}/{args.num_epochs} completed.")
        p = "||".join([f"{k}:{v}" for k, v in train_metrics.items()])
        print(
            f"train loss:{train_loss}||InfoNCELoss: {train_info_nce_loss}||CrossEntropyLoss: {train_cross_entropy_loss}||{p}"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": trained_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            f"{save_path}/{model_name}_latest.pth",
        )

        # eval
        test_loss, test_info_nce_loss, test_cross_entropy_loss, _, test_metrics = (
            train_model(
                model=clip_model,
                dataloader=test_dataloader,
                optimizer=optimizer,
                criterion=criterion,
                mode="test",
                cluster_emb=cluster_model.cluster_emb,
            )
        )
        p = "||".join([f"{k}:{v}" for k, v in test_metrics.items()])
        print(
            f"test loss:{test_loss}||InfoNCELoss: {test_info_nce_loss}||CrossEntropyLoss: {test_cross_entropy_loss}||{p}"
        )

        # record training loss and metrics
        tb_metric_weight(
            writer=writer,
            train_plot_data=[
                train_metrics,
                train_loss,
                train_info_nce_loss,
                train_cross_entropy_loss,
            ],
            test_plot_data=[
                test_metrics,
                test_loss,
                test_info_nce_loss,
                test_cross_entropy_loss,
            ],
            epoch=epoch,
        )

        if train_loss <= min_train_loss:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": trained_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                    "Acc@1": train_metrics["Acc@1"],
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                f"{save_path}/{model_name}_train_min_loss.pth",
            )
            min_train_loss = train_loss
            print(f"saved train model at epoch {epoch} for min loss")

        if train_metrics["Acc@1"] >= max_train_acc:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": trained_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                    "Acc@1": train_metrics["Acc@1"],
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                f"{save_path}/{model_name}_train_max_acc.pth",
            )
            max_train_acc = train_metrics["Acc@1"]
            print(f"saved train model at epoch {epoch} for max acc")

        if test_loss <= min_test_loss:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": trained_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": test_loss,
                    "Acc@1": train_metrics["Acc@1"],
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                f"{save_path}/{model_name}_test_min_loss.pth",
            )
            min_test_loss = test_loss
            print(f"saved test model at epoch {epoch} for min loss")

        if test_metrics["Acc@1"] >= max_test_acc:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": trained_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": test_loss,
                    "Acc@1": test_metrics["Acc@1"],
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                f"{save_path}/{model_name}_test_max_acc.pth",
            )
            max_test_acc = (test_metrics["Acc@1"],)
            print(f"saved test model at epoch {epoch} for max acc")
    writer.close()
exit(0)

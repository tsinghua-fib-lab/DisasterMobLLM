import torch.distributed
import torch.utils
import logging
import torch.utils
from main_model import *
from CLIP_model import *
from RAG_model import *
from LLM_util import *
from transformers import LlamaForCausalLM, LlamaModel
from torch.optim import AdamW
import argparse
from datetime import datetime
from collections import OrderedDict
import warnings
import time
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--prefix_length", type=int, default=20)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--multi_gpu", type=str2bool, default=False)
parser.add_argument("--use_LLM", type=str, default="meta-llama/Llama-3.2-8B")
parser.add_argument("--target_city", type=str, default="zhongshan_01")
parser.add_argument("--path", type=str, default="/your_path/Mob_data/datas")
parser.add_argument("--LLM_path", type=str, default="/your_path/LLMs")
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--gpu_num", type=int, default=1)
parser.add_argument("--keep_training", type=str2bool, default=False)
parser.add_argument("--generate_answer", type=str2bool, default=True)
args = parser.parse_args()

if __name__ == "__main__":

    # init statics
    train, test = get_train_test_set(target_city=args.target_city)
    model_name, hidden_dim = get_modelname_and_hiddim(args.use_LLM)

    # init tensorboard
    log_dir = f"{args.path}/util_datas/tensorboard_logs/{model_name}/IntentionPred"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # set save path
    if args.generate_answer:
        save_path = f"{args.path}/util_datas/prefix_model/{model_name}_backup"
    else:
        save_path = f"{args.path}/util_datas/prefix_model/{model_name}"
    os.makedirs(save_path, exist_ok=True)

    # init logging
    log_file = f"{save_path}/run_log.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = setup_logging(log_file)
    logger.info("Starting the training process...")

    # init model
    model = LlamaForCausalLM.from_pretrained(args.use_LLM, cache_dir=args.LLM_path)
    text_vocab = (
        model.get_input_embeddings().weight
        if isinstance(model, LlamaForCausalLM)
        else model.embed_tokens.weight
    )
    text_tokenizor = AutoTokenizer.from_pretrained(args.use_LLM)
    # init dataset
    train_dataset = MainDataset(
        path=args.path,
        datasets=train,
        mode="train",
        model_name=model_name,
        hidden_dim=hidden_dim,
        tokenizer=text_tokenizor,
        generate_answer=True if args.generate_answer else False,
    )
    test_dataset = MainDataset(
        path=args.path,
        datasets=test,
        mode="test",
        model_name=model_name,
        hidden_dim=hidden_dim,
        tokenizer=text_tokenizor,
        scaler=train_dataset.scaler,
        intention_scaler=train_dataset.intention_scaler,
        generate_answer=True if args.generate_answer else False,
    )
    if args.multi_gpu:
        # Multi-GPU
        logger.info("Using multi GPU")
        torch.distributed.init_process_group(backend="nccl")
        logger.info(f"Local rank: {args.local_rank}")
        torch.cuda.set_device(args.local_rank)
        logger.info(f"Local rank {args.local_rank} finished setting up device")
        prefix_model = nn.parallel.DistributedDataParallel(
            Prefix_LLM(
                text_tokenizer=text_tokenizor,
                llm_model=model,
                prefix_length=args.prefix_length,
                hidden_dim=hidden_dim,
            ).cuda(args.local_rank),
            device_ids=[args.local_rank],
            find_unused_parameters=True,
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, sampler=test_sampler
        )
        for param in prefix_model.module.llm_model.parameters():
            param.requires_grad = False  # 冻结所有参数
        for param in prefix_model.module.prefix.parameters():
            param.requires_grad = True  # 只解冻前缀参数
        optimizer = AdamW(
            list(prefix_model.module.prefix.parameters()),
            lr=5e-3,
        )
    else:
        logger.info("Using single GPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("use single GPU")
        prefix_model = Prefix_LLM(
            text_tokenizer=text_tokenizor,
            llm_model=model,
            prefix_length=args.prefix_length,
            hidden_dim=hidden_dim,
        ).cuda()
        if (args.keep_training or args.generate_answer) and os.path.exists(
            f"{save_path}/{model_name}_test_max_acc.pth"
        ):
            state_dict = torch.load(
                f"{save_path}/{model_name}_test_max_acc.pth",
                map_location="cuda:0",
            )["model_state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k  # 移除 `module.`
                new_state_dict[name] = v
            prefix_model.prefix.load_state_dict(new_state_dict)
            prefix_model = prefix_model.cuda()
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)
        for param in prefix_model.llm_model.parameters():
            param.requires_grad = False  # 冻结所有参数
        for param in prefix_model.prefix.parameters():
            param.requires_grad = True  # 只解冻前缀参数
        optimizer = AdamW(
            list(prefix_model.prefix.parameters()),
            lr=5e-3,
        )

    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    min_train_loss = 1e8
    min_test_loss = 1e8
    max_train_acc = 0
    max_test_acc = 0

    # start training
    for epoch in trange(args.num_epochs):
        logger.info(f"Epoch {epoch}/{args.num_epochs} started")
        if not args.generate_answer:
            train_loss, train_metrics, prefix_model, optimizer = train_test_LLM_model(
                args=args,
                prefix_model=prefix_model,
                dataloader=train_dataloader,
                dataset=train_dataset,
                mode="train",
                loss_fn=loss_fn,
                optimizer=optimizer,
                model_name=model_name,
                logger=logger,
            )

            info_str = f"【GPU{args.local_rank}】Epoch {epoch}/{args.num_epochs} - Train Loss: {train_loss:.4f}, "
            for k, v in train_metrics.items():
                info_str += f"{k}: {v}, "
            logger.info(info_str)
            if args.local_rank == 0:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": (
                            prefix_model.module.state_dict()
                            if args.multi_gpu
                            else prefix_model.state_dict()
                        ),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": train_loss,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    f"{save_path}/latest.pt",
                )
        else:
            print(1)
            # train_loss, train_metrics, prefix_model, optimizer = train_test_LLM_model(
            #     args=args,
            #     prefix_model=prefix_model,
            #     dataloader=train_dataloader,
            #     dataset=train_dataset,
            #     mode="train",
            #     loss_fn=loss_fn,
            #     optimizer=optimizer,
            #     model_name=model_name,
            #     logger=logger,
            # )
        test_loss, test_metrics, _, _ = train_test_LLM_model(
            args=args,
            prefix_model=prefix_model,
            dataloader=test_dataloader,
            dataset=test_dataset,
            mode="test",
            loss_fn=loss_fn,
            optimizer=optimizer,
            model_name=model_name,
            logger=logger,
        )
        info_str = f"【GPU{args.local_rank}】Epoch {epoch}/{args.num_epochs} - Test Loss: {test_loss:.4f}, "
        for k, v in test_metrics.items():
            info_str += f"{k}: {v}, "
        logger.info(info_str)
        if args.local_rank == 0:
            tb_metric_weight(
                writer=writer,
                train_plot_data=[train_metrics, train_loss],
                test_plot_data=[test_metrics, test_loss],
                epoch=epoch,
            )

        if train_loss <= min_train_loss and args.local_rank == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": (
                        prefix_model.module.prefix.state_dict()
                        if args.multi_gpu
                        else prefix_model.prefix.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                    "Acc@Intention": train_metrics["Acc@Intention"],
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                f"{save_path}/{model_name}_train_min_loss.pth",
            )
            min_train_loss = train_loss
            logger.info(f"Saved train model at epoch {epoch} for min loss")

        if train_metrics["Acc@Intention"] >= max_train_acc and args.local_rank == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": (
                        prefix_model.module.prefix.state_dict()
                        if args.multi_gpu
                        else prefix_model.prefix.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                    "Acc@Intention": train_metrics["Acc@Intention"],
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                f"{save_path}/{model_name}_train_max_acc.pth",
            )
            max_train_acc = train_metrics["Acc@Intention"]
            logger.info(f"Saved train model at epoch {epoch} for max acc")

        if test_metrics["Acc@Intention"] >= max_test_acc and args.local_rank == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": (
                        prefix_model.module.prefix.state_dict()
                        if args.multi_gpu
                        else prefix_model.prefix.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": test_loss,
                    "Acc@Intention": test_metrics["Acc@Intention"],
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                f"{save_path}/{model_name}_test_max_acc.pth",
            )
            max_test_acc = (test_metrics["Acc@Intention"],)
            logger.info(f"Saved test model at epoch {epoch} for max acc")
    writer.close()

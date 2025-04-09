import torch.distributed
import torch.utils
import torch.utils
from main_model import *
from CLIP_model import *
from RAG_model import *
from LLM_util import *
from transformers import LlamaForCausalLM, LlamaModel
from torch.optim import AdamW
import argparse
from NNPM_model import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--prefix_length", type=int, default=20)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--multi_gpu", type=str2bool, default=False)
parser.add_argument("--use_LLM", type=str, default="meta-llama/Llama-3.2-8B")
parser.add_argument("--LLM_path", type=str, default="/your_path/LLMs")
parser.add_argument("--city", type=str, default="zhongshan_01")
parser.add_argument("--normal_model", type=str, default="DeepMove")
parser.add_argument("--path", type=str, default="/your_path/Mob_data/datas")
args = parser.parse_args()

if __name__ == "__main__":
    # init statics
    train, test = get_train_test_set(target_city=args.city)
    model_name, hidden_dim = get_modelname_and_hiddim(args.use_LLM)

    # init tensorboard
    log_dir = f"{args.path}/util_datas/tensorboard_logs/{model_name}/RegionProj"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # set save path
    save_path = f"{args.path}/util_datas/NNPMs/{model_name}"
    os.makedirs(save_path, exist_ok=True)

    # init logging
    log_file = f"{save_path}/run_log.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = setup_logging(log_file)
    logger.info("Starting the training process...")

    model = LlamaModel.from_pretrained(args.use_LLM, cache_dir=args.LLM_path)
    text_vocab = model.embed_tokens.weight
    text_tokenizor = AutoTokenizer.from_pretrained(args.use_LLM)  # LLM的tokenizor
    train_dataset = NNPMDataset(
        path=args.path,
        city=args.city,
        mode="train",
        model_name=model_name,
        normal_model=args.normal_model,
    )
    test_dataset = NNPMDataset(
        path=args.path,
        city=args.city,
        mode="test",
        model_name=model_name,
        normal_model=args.normal_model,
    )
    region_num = train_dataset.grid
    if args.multi_gpu:
        # Multi-GPU
        print("use multi GPU")
        n_gpus = 2
        torch.distributed.init_process_group(
            "nccl", world_size=n_gpus, rank=args.local_rank
        )
        torch.cuda.set_device(args.local_rank)
        pred_model = nn.parallel.DistributedDataParallel(
            NNPM(
                text_tokenizer=text_tokenizor,
                llm_model=model,
                prefix_length=args.prefix_length,
                region_num=region_num,
                hidden_dim=hidden_dim,
                model_path=f"{args.path}/util_datas/prefix_model/{model_name}/best.pt",
                cluster_emb_path=f"{args.path}/util_datas/intention_emb/{model_name}.json",
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
        for param in pred_model.module.prefix_model.parameters():
            param.requires_grad = False  # 冻结所有参数
        for param in pred_model.module.region_proj.parameters():
            param.requires_grad = True  # 只解冻前缀参数
        optimizer = AdamW(
            list(pred_model.module.region_proj.parameters()),
            lr=1e-4,
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print("use single GPU")
        pred_model = NNPM(
            text_tokenizor=text_tokenizor,
            llm_model=model,
            prefix_length=args.prefix_length,
            region_num=region_num,
            hidden_dim=hidden_dim,
            model_path=f"{args.path}/util_datas/prefix_model/{model_name}/best.pt",
            cluster_emb_path=f"{args.path}/util_datas/intention_emb/{model_name}.json",
        ).cuda(args.local_rank)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)
        for param in pred_model.prefix_model.parameters():
            param.requires_grad = False  # 冻结所有参数
        for param in pred_model.region_proj.parameters():
            param.requires_grad = True  # 只解冻前缀参数
        optimizer = AdamW(
            list(pred_model.region_proj.parameters()),
            lr=1e-3,
        )

    loss_fn = nn.CrossEntropyLoss()
    min_loss = 1e8
    for epoch in trange(args.num_epochs):
        # train model
        train_loss, train_metric, pred_model, optimizer = train_test_NNPM_model(
            args=args,
            pred_model=pred_model,
            dataloader=train_dataloader,
            dataset=train_dataset,
            mode="train",
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": (
                    pred_model.module.state_dict()
                    if args.multi_gpu
                    else pred_model.state_dict()
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
            },
            f"{save_path}/{args.normal_model}_latest.pt",
        )
        # test model
        test_loss, test_metric, _, _ = train_test_NNPM_model(
            args=args,
            pred_model=pred_model,
            dataloader=test_dataloader,
            dataset=test_dataset,
            mode="test",
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        print(
            f"Training Statics for epoch {epoch}=============>Loss:{train_loss} "
            + " ".join([f"{k}:{np.mean(v)}" for k, v in train_metric.items()])
        )
        print(
            f"Testing Statics for epoch {epoch}=============>Loss:{test_loss} "
            + " ".join([f"{k}:{np.mean(v)}" for k, v in test_metric.items()])
        )
        if test_loss < min_loss and args.local_rank == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        pred_model.module.state_dict()
                        if args.multi_gpu
                        else pred_model.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": test_loss.item(),
                },
                f"{save_path}/{args.normal_model}_best.pt",
            )
            min_loss = test_loss

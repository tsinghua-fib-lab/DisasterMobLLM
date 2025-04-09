import torch.distributed
import torch.utils
import torch.utils
from main_model import *
from CLIP_model import *
from RAG_model import *
from LLM_util import *
from transformers import LlamaForCausalLM, LlamaModel
import argparse
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--input_dim", type=int, default=111)
parser.add_argument("--intention_dim", type=int, default=20)
parser.add_argument("--traj_pad_length", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--attn_dim", type=int, default=512)
parser.add_argument("--have_db", type=str2bool, default=True)
parser.add_argument("--proto_dim", type=int, default=1024)
parser.add_argument("--retrieval_k", type=int, default=3)
parser.add_argument("--use_LLM", type=str, default="meta-llama/Llama-3.2-8B")
parser.add_argument("--target_city", type=str, default="zhongshan_01")
parser.add_argument("--path", type=str, default="/your_path/Mob_data/datas")
parser.add_argument("--LLM_path", type=str, default="/your_paths/LLMs")
parser.add_argument(
    "--mode",
    type=str,
    default="train",
)
parser.add_argument("--use_multi_CPU", type=str2bool, default=False)
args = parser.parse_args()


def main():
    if args.use_multi_CPU:
        multiprocessing.set_start_method("spawn", force=True)
    train, test = get_train_test_set(target_city=args.target_city)
    model_name, hidden_dim = get_modelname_and_hiddim(args.use_LLM)
    print(f"Use LLM: {model_name}")
    db_path = f"{args.path}/util_datas/storage/{model_name}"
    os.makedirs(db_path, exist_ok=True)
    save_path = f"{args.path}/util_datas/traj_dataset/{model_name}/{args.mode}"
    print(f"save_path:{save_path}")
    os.makedirs(save_path, exist_ok=True)
    model = AutoModel.from_pretrained(args.use_LLM, cache_dir=args.LLM_path)
    text_vocab = model.embed_tokens.weight
    cluster_model = Cluster(
        datasets=train, max_cluster_num=50, have_c=True, path=args.path
    )
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
    clip_path = f"{args.path}/util_datas/CLIP_checkpoints/{model_name}/{model_name}_test_max_acc.pth"
    states = torch.load(clip_path)
    clip_model.load_state_dict(torch.load(clip_path)["model_state_dict"])
    print(
        f"Loaded trained CLIP: Loss-{states['loss']} || Acc@1-{states['Acc@1']} || time-{states['time']}"
    )
    clip_model.eval()
    text_tokenizor = clip_model.text_tokenizor  # LLM的tokenizor
    traj_tokenizor = clip_model.traj_tokenizor  # 来自预训练的CLIP
    traj_encoder = clip_model.trajectory_encoder  # 来自预训练的CLIP
    traj_RAG = TrajRAG(
        db_path=db_path,
        datasets=train,
        trajectory_tokenizor=traj_tokenizor,
        trajectory_encoder=traj_encoder,
        have_db=args.have_db,
    )  # CLIP训练完后要重新生成一下
    if not args.have_db:
        print(
            "generate completed, set have_db in args as True and run this code again!"
        )
        exit(0)
    if args.mode == "train":
        dataset = TrajDataset(
            traj_RAG=traj_RAG,
            datasets=train,
            traj_pad_length=args.traj_pad_length,
            traj_encoder=traj_encoder,
            retrieval_k=args.retrieval_k,
            clip_model=clip_model,
            cluster_emb=cluster_model.cluster_emb,
            hidden_dim=hidden_dim,
            text_tokenizer=text_tokenizor,
            path=args.path,
            model_name=model_name,
            immob_emb=clip_model.immob_emb.detach().cpu().numpy().tolist()[0],
            cluster_model=cluster_model,
        )
    else:
        dataset = TrajDataset(
            traj_RAG=traj_RAG,
            datasets=test,
            traj_pad_length=args.traj_pad_length,
            traj_encoder=traj_encoder,
            retrieval_k=args.retrieval_k,
            clip_model=clip_model,
            cluster_emb=cluster_model.cluster_emb,
            hidden_dim=hidden_dim,
            text_tokenizer=text_tokenizor,
            path=args.path,
            model_name=model_name,
            immob_emb=clip_model.immob_emb.detach().cpu().numpy().tolist()[0],
            cluster_model=cluster_model,
        )
    if args.use_multi_CPU:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
        )
    index = 0
    for (
        reg_seq,
        reg_intention_emb_seq,
        intention_seq,
        next_intention_emb,
        next_reg,
        next_intention,
        disaster_level,
        ref_intention_emb_seq,
        tokens,
        masks,
        change_or_not,
        change_to_immob,
        token_type,
    ) in tqdm(dataloader):
        dataset = []
        for batch in range(args.batch_size):
            dataset = {
                "reg_seq": reg_seq[batch].detach().cpu().numpy().tolist(),
                "reg_intention_emb_seq": reg_intention_emb_seq[batch]
                .detach()
                .cpu()
                .numpy()
                .tolist(),
                "intention_seq": intention_seq[batch].detach().cpu().numpy().tolist(),
                "next_intention_emb": next_intention_emb[batch]
                .detach()
                .cpu()
                .numpy()
                .tolist(),
                "next_reg": next_reg[batch].detach().cpu().numpy().tolist(),
                "next_intention": next_intention[batch].detach().cpu().numpy().tolist(),
                "disaster_level": disaster_level[batch],
                "ref_intention_emb_seq": [
                    x[batch].detach().cpu().numpy().tolist()
                    for x in ref_intention_emb_seq
                ],
                "tokens": [x[batch].detach().cpu().numpy().tolist() for x in tokens],
                "masks": [x[batch].detach().cpu().numpy().tolist() for x in masks],
                "change_or_not": change_or_not[batch].detach().cpu().numpy().tolist(),
                "change_to_immob": change_to_immob[batch]
                .detach()
                .cpu()
                .numpy()
                .tolist(),
                "token_type": [
                    x[batch].detach().cpu().numpy().tolist() for x in token_type
                ],
            }
            with open(
                f"{save_path}/part_{index}.json",
                "w",
            ) as f:
                json.dump(dataset, f)
            index += 1


if __name__ == "__main__":
    main()

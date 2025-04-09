from CLIP_model import *
from RAG_model import *
from utils import *
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from main_model import *
from LLM_util import *
from collections import deque
from transformers import LogitsProcessorList
import random


class TrajRAG:
    def __init__(
        self,
        db_path,
        datasets,
        trajectory_tokenizor,
        trajectory_encoder,
        have_db=True,
    ):
        self.embedding = IntentionEmbedding(
            trajectory_encoder=trajectory_encoder,
            trajectory_tokenizor=trajectory_tokenizor,
        )
        if have_db:
            # 保存数据库之后
            self.vector = VectorStore()
            self.vector.load_vector(db_path)  # 加载本地的数据库
            print(f"loaded database from {db_path}")
        else:
            print("no existing db, creating")
            # 获得data目录下的所有文件内容并分割
            docs = ReadFiles(
                path="/your_path/Mob_data/datas", dataset=datasets
            ).get_content()

            self.vector = VectorStore(docs)
            self.vector.get_vector(EmbeddingModel=self.embedding)
            self.vector.persist(path=db_path)
            print(f"created database in {db_path}")

    def retrieval_traj(self, traj, disaster_level):
        content = self.vector.query(
            traj, disaster_level, EmbeddingModel=self.embedding, k=3
        )
        return content


class TrajDataset(Dataset):
    def __init__(
        self,
        traj_RAG,
        datasets,
        traj_pad_length,
        traj_encoder,
        retrieval_k,
        clip_model,
        cluster_emb,
        hidden_dim,
        text_tokenizer,
        path,
        model_name,
        immob_emb,
        cluster_model,
        city_immob_indexs={},
    ):
        self.dataset = datasets
        self.immob_emb = immob_emb
        self.traj_pad_length = traj_pad_length
        self.text_tokenizer = text_tokenizer
        self.path = path
        self.trajectories = []
        self.traj_encoder = traj_encoder
        self.retrieval_k = retrieval_k
        self.clip_model = clip_model
        self.cluster_emb = {}
        self.hidden_dim = hidden_dim
        immob_index = len(cluster_model.cluster_emb)
        for dataset in self.dataset:
            city, mode = dataset.split("-")
            if not city in city_immob_indexs.keys():
                city_immob_indexs[city] = 0
            not_immob_index = city_immob_indexs[city]
            with open(
                f"{path}/util_datas/CLIP_input/{city}.json",
                "r",
            ) as f:
                tca_data = json.load(f)
            if os.path.exists(
                f"{self.path}/util_datas/main_dataset/{city}/{mode}.json"
            ):
                with open(
                    f"{self.path}/util_datas/main_dataset/{city}/{mode}.json",
                    "r",
                ) as f:
                    data = json.load(f)
                for tid, traj in enumerate(data):
                    intention_seq = []
                    intention_emb_seq = []
                    for switch in traj["st_emb_seq"]:
                        if not switch[64] == 0:
                            intention_emb = tca_data[not_immob_index]
                            intention, intention_index = cluster_model.get_cluster(
                                intention_emb
                            )
                            not_immob_index += 1
                            intention_seq.append(intention_index)
                            intention_emb_seq.append(intention)
                        else:
                            intention_seq.append(immob_index)
                            intention_emb_seq.append(np.zeros(20).tolist())
                    data[tid]["intention_seq"] = intention_seq
                    data[tid]["intention_emb_seq"] = intention_emb_seq
                self.trajectories.extend(data)
            city_immob_indexs[city] = not_immob_index
        self.TrajRAG = traj_RAG
        cluster_emb[str(immob_index)] = (
            clip_model.immob_emb.detach().cpu().numpy().reshape(-1).tolist()
        )
        emb = torch.tensor(
            np.array(list(cluster_emb.values())),
            dtype=torch.float32,
            requires_grad=False,
        )
        fake_traj = torch.tensor(
            np.zeros((1, 1, 111)), dtype=torch.float32, requires_grad=False
        )
        _, intention_embedding, logits = self.clip_model(fake_traj.cuda(), emb.cuda())
        intention_embedding = intention_embedding.detach().cpu().numpy().tolist()
        for index in range(len(intention_embedding)):
            self.cluster_emb[index] = intention_embedding[index]
        with open(f"{path}/util_datas/intention_emb/{model_name}.json", "w") as f:
            json.dump(self.cluster_emb, f)
        # self.cluster_emb[-1] = np.zeros(self.hidden_dim).tolist()
        # print(1)
        self.cluster_intention_embedding = intention_embedding

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj_info = self.trajectories[idx]
        reg_seq = traj_info["grid_seq"]
        intention_emb_seq, intention_seq = self.get_intention_and_emb_seq(traj_info)
        disaster_level = traj_info["disaster_level"]
        similar_trajs = self.TrajRAG.retrieval_traj(traj_info, disaster_level)
        ref_reg_seq = [x["grid_seq"] for x in similar_trajs]
        ref_intention_emb_seq_temp = [
            self.get_intention_and_emb_seq(x)[0] for x in similar_trajs
        ]
        ref_intention_emb_seq = []
        for x in ref_intention_emb_seq_temp:
            if self.traj_pad_length - x.shape[0] >= 0:
                x = np.pad(
                    np.array(x),
                    ((0, self.traj_pad_length - x.shape[0]), (0, 0)),
                    "constant",
                    constant_values=-1,
                )
            else:
                x = np.array(x)[: self.traj_pad_length]
            ref_intention_emb_seq.append(x)
        if self.traj_pad_length - len(reg_seq[:-1]) >= 0:
            ret_reg_seq = np.pad(
                np.array(reg_seq[:-1]),
                (0, self.traj_pad_length - len(reg_seq[:-1])),
                "constant",
                constant_values=-1,
            )
        else:
            ret_reg_seq = np.array(reg_seq[:-1])[: self.traj_pad_length]
        if self.traj_pad_length - len(intention_seq[:-1]) >= 0:
            ret_intention_seq = np.pad(
                np.array(intention_seq[:-1]),
                (0, self.traj_pad_length - len(intention_seq[:-1])),
                "constant",
                constant_values=-1,
            )
        else:
            ret_intention_seq = np.array(intention_seq[:-1])[: self.traj_pad_length]
        if self.traj_pad_length - len(intention_emb_seq[:-1]) >= 0:
            ret_reg_intention_emb_seq = np.pad(
                np.array(intention_emb_seq[:-1]),
                ((0, self.traj_pad_length - len(intention_emb_seq[:-1])), (0, 0)),
                "constant",
                constant_values=-1,
            )
        else:
            ret_reg_intention_emb_seq = np.array(intention_emb_seq[:-1])[
                : self.traj_pad_length
            ]
        ret_next_reg = np.array(reg_seq[-1])
        # ret_next_intention_emb = np.array(intention_emb_seq[-1])
        # ret_next_intention = np.array(intention_seq[-1])
        ret_next_intention = np.array(traj_info["intention_seq"][-1])
        ret_next_intention_emb = np.array(
            self.cluster_intention_embedding[int(ret_next_intention)]
        )
        ret_disaster_level = disaster_level
        if int(ret_next_intention) == int(intention_seq[-1]):
            change_or_not = 0  # no need to change answer
            change_to_immob = 0
        else:
            # need to change answer
            change_or_not = 1
            if int(ret_next_intention) == 15:
                # need to change to immob
                change_to_immob = 1
            else:
                # need to change to other
                change_to_immob = 0
        tokens, masks, token_type = get_tokenized_prompt(
            self.text_tokenizer,
            ret_reg_intention_emb_seq,
            ref_intention_emb_seq,
            disaster_level,
            similar_trajs,
            ret_next_intention_emb,
            self.cluster_intention_embedding,
        )
        return (
            ret_reg_seq,
            ret_reg_intention_emb_seq,
            ret_intention_seq,
            ret_next_intention_emb,
            ret_next_reg,
            ret_next_intention,
            ret_disaster_level,
            ref_intention_emb_seq,
            tokens,
            masks,
            torch.tensor(change_or_not, dtype=int),
            torch.tensor(change_to_immob, dtype=int),
            token_type,
        )

    def get_intention_and_emb_seq(self, traj_info):
        immob_emb = self.immob_emb
        intention_emb_seq = []
        for point in traj_info["st_emb_seq"]:
            encoded_input = torch.tensor(
                np.expand_dims(np.expand_dims(point, axis=0), axis=0),
                dtype=torch.float32,
            ).cuda()
            if not point[64] == 0:
                with torch.no_grad():
                    model_output, logits = self.traj_encoder(encoded_input)
                intention_emb_seq.append(model_output.cpu().numpy().tolist()[0])
            else:
                intention_emb_seq.append(np.zeros(self.hidden_dim).tolist())
        cluster_emb = torch.tensor(
            np.array(list(self.cluster_emb.values())), dtype=torch.float32
        )
        intention_emb_seq = torch.tensor(
            np.array(intention_emb_seq), dtype=torch.float32
        )
        # 计算余弦相似度
        similarity = cosine_similarity(
            F.normalize(intention_emb_seq, dim=1), F.normalize(cluster_emb, dim=1)
        )
        max_similarity_indices = np.argmax(similarity, axis=1)
        intention_seq = max_similarity_indices.tolist()
        for index, x in enumerate(intention_emb_seq.detach().cpu().numpy().tolist()):
            if not np.all(np.array(x) == 0):
                intention_emb_seq[index] = cluster_emb[intention_seq[index]]
            else:
                intention_emb_seq[index] = cluster_emb[-1]
        return intention_emb_seq.detach().cpu().numpy(), intention_seq


class PrefixEmbedding(nn.Module):
    def __init__(self, prefix_length, hidden_size):
        super(PrefixEmbedding, self).__init__()
        self.prefix_embedding = nn.Parameter(torch.randn(4, prefix_length, hidden_size))

    def forward(self, dis_kind):
        return self.prefix_embedding[dis_kind, :, :].unsqueeze(0)


class FixedTokenLogitsProcessor:
    def __init__(self, fixed_token_ids):
        self.fixed_token_ids = set(fixed_token_ids)

    def __call__(self, input_ids, scores):
        # 将不在固定token列表中的token的概率设为负无穷
        for i in range(scores.shape[1]):
            if i not in self.fixed_token_ids:
                scores[:, i] = -float("inf")
        return scores


class Prefix_LLM(nn.Module):
    def __init__(
        self,
        text_tokenizer,
        llm_model,
        prefix_length,
        hidden_dim,
    ):
        super(Prefix_LLM, self).__init__()
        self.text_tokenizer = text_tokenizer
        self.immob_ember = ImmobilityEmbedding(text_tokenizer, llm_model)
        vocab_num = text_tokenizer.vocab_size
        self.llm_model = llm_model
        self.prefix = PrefixEmbedding(prefix_length, hidden_dim)
        self.hidden_dim = hidden_dim
        self.dis_map = {}
        i = 0
        for disaster in [
            "no disaster",
            "minor disaster",
            "general disaster",
            "severe disaster",
        ]:
            self.dis_map[disaster] = i
            i += 1
        # 定义固定的token列表
        fixed_tokens = ["yes", "no"]
        intention_tokens = [str(x) for x in range(16)]
        # 获取这些token的ID
        fixed_token_ids = text_tokenizer(
            fixed_tokens, add_special_tokens=False
        ).input_ids
        intention_token_ids = text_tokenizer(
            intention_tokens, add_special_tokens=False
        ).input_ids
        fixed_token_ids = [id[0] for id in fixed_token_ids]
        intention_token_ids = [id[0] for id in intention_token_ids]
        self.logits_processor_12 = LogitsProcessorList(
            [FixedTokenLogitsProcessor(fixed_token_ids)]
        )
        self.logits_processor_3 = LogitsProcessorList(
            [FixedTokenLogitsProcessor(intention_token_ids)]
        )

    def forward(self, disaster_level, tokens, masks, batch_size, local_rank):
        input_embedding, attention_mask = self.get_prompt_info(
            tokens=tokens, masks=masks, local_rank=local_rank
        )
        prefix = torch.cat(
            [self.prefix(self.dis_map[x]) for x in disaster_level],
            dim=0,
        ).to(local_rank)
        input_embeds = torch.cat([prefix, input_embedding], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones(
                    (prefix.shape[0], prefix.shape[1]),
                    dtype=torch.long,
                    device=local_rank,
                ),
                attention_mask,
            ],
            dim=1,
        )
        llm_output = self.llm_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        )

        logits = llm_output.logits
        classes = torch.argmax(logits, dim=2)
        # outputs = self.text_tokenizer.batch_decode(classes)
        return classes, logits

    def add_token(self, input_embeds, attention_mask, token):
        if isinstance(token, str):
            tk = self.text_tokenizer.encode(
                token, return_tensors="pt", add_special_tokens=False
            ).to(input_embeds.device)
            input_embeds = torch.cat(
                [
                    input_embeds,
                    self.llm_model.get_input_embeddings()(tk).repeat(
                        input_embeds.shape[0], 1, 1
                    ),
                ],
                dim=1,
            )
            attention_mask = torch.cat(
                [
                    torch.ones(
                        (attention_mask.shape[0], tk.shape[0]),
                        dtype=torch.long,
                        device=input_embeds.device,
                    ),
                    attention_mask,
                ],
                dim=1,
            )
        else:
            input_embeds = torch.cat(
                [
                    input_embeds,
                    self.llm_model.get_input_embeddings()(token),
                ],
                dim=1,
            )
            attention_mask = torch.cat(
                [
                    torch.ones(
                        (attention_mask.shape[0], token.shape[1]),
                        dtype=torch.long,
                        device=input_embeds.device,
                    ),
                    attention_mask,
                ],
                dim=1,
            )
        return input_embeds, attention_mask

    def generate_text(self, disaster_level, tokens, masks, local_rank):

        input_embedding, attention_mask = self.get_prompt_info(
            tokens=tokens, masks=masks, local_rank=local_rank
        )
        prefix_indices = torch.tensor(
            [self.dis_map[x] for x in disaster_level], device=local_rank
        )
        prefix = self.prefix(prefix_indices).squeeze(0)
        input_embeds = torch.cat([prefix, input_embedding], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones(
                    (prefix.shape[0], prefix.shape[1]),
                    dtype=torch.long,
                    device=local_rank,
                ),
                attention_mask,
            ],
            dim=1,
        )
        with torch.no_grad():
            input_embeds, attention_mask = self.add_token(
                input_embeds, attention_mask, "["
            )
            for i in range(5):
                try:
                    outputs = [
                        self.text_tokenizer.encode(
                            "[", return_tensors="pt", add_special_tokens=False
                        )
                        .repeat(input_embeds.shape[0], 1)
                        .to(input_embeds.device)
                    ]
                    outputs_1 = self.llm_model.generate(
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask,
                        max_new_tokens=1,
                        temperature=2.0,  # 调整温度参数
                        do_sample=True,  # 使用采样生成
                        logits_processor=self.logits_processor_12,
                    )
                    input_embeds, attention_mask = self.add_token(
                        input_embeds, attention_mask, outputs_1
                    )
                    input_embeds, attention_mask = self.add_token(
                        input_embeds, attention_mask, ","
                    )
                    outputs.append(outputs_1)
                    outputs.append(
                        self.text_tokenizer.encode(
                            ",", return_tensors="pt", add_special_tokens=False
                        )
                        .repeat(input_embeds.shape[0], 1)
                        .to(input_embeds.device)
                    )
                    outputs_2 = self.llm_model.generate(
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask,
                        max_new_tokens=1,
                        temperature=2.0,  # 调整温度参数
                        do_sample=True,  # 使用采样生成
                        logits_processor=self.logits_processor_12,
                    )
                    input_embeds, attention_mask = self.add_token(
                        input_embeds, attention_mask, outputs_2
                    )
                    input_embeds, attention_mask = self.add_token(
                        input_embeds, attention_mask, ","
                    )
                    outputs.append(outputs_2)
                    outputs.append(
                        self.text_tokenizer.encode(
                            ",", return_tensors="pt", add_special_tokens=False
                        )
                        .repeat(input_embeds.shape[0], 1)
                        .to(input_embeds.device)
                    )
                    outputs_3 = self.llm_model.generate(
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask,
                        max_new_tokens=1,
                        temperature=2.0,  # 调整温度参数
                        do_sample=True,  # 使用采样生成
                        logits_processor=self.logits_processor_3,
                    )
                    outputs.append(outputs_3)
                    outputs.append(
                        self.text_tokenizer.encode(
                            "]", return_tensors="pt", add_special_tokens=False
                        )
                        .repeat(input_embeds.shape[0], 1)
                        .to(input_embeds.device)
                    )
                    outputs = torch.concatenate(outputs, dim=1)
                    break
                except:
                    print(f"GENERATE ERROR Retrying for {i} times")

            if i == 4:
                exit(0)
        return outputs

    def get_prompt_info(self, tokens, masks, local_rank):
        batch_size = masks[0].shape[0]
        input_embedding = [[] for _ in range(batch_size)]
        attention_mask = [[] for _ in range(batch_size)]
        for index in range(len(tokens)):
            token = tokens[index]
            mask = masks[index]
            try:
                if mask[0]:
                    # print(self.text_tokenizer.batch_decode(token))
                    embeddings = (
                        self.llm_model.embed_tokens(token.squeeze(1))
                        if isinstance(self.llm_model, LlamaModel)
                        else self.llm_model.get_input_embeddings()(token.squeeze(1))
                    )
                    for x in range(batch_size):
                        input_embedding[x].append(
                            embeddings[x].reshape(-1, self.hidden_dim)
                        )
                        attention_mask[x].append(torch.ones(token.shape[1]))
                else:
                    for x in range(batch_size):
                        if torch.sum(token[x]) == 0:
                            immob = self.immob_ember().cuda(local_rank)
                            # input_embedding[x].append(immob.to(self.device))
                            # attention_mask[x].append(torch.ones(1, device=self.device))
                            input_embedding[x].append(immob)
                            attention_mask[x].append(torch.ones(1))
                        elif torch.all(token[x] == -1):
                            # input_embedding[x].append(
                            #     torch.zeros(len(token[x])).unsqueeze(0).to(self.device)
                            # )
                            # attention_mask[x].append(torch.zeros(1, device=self.device))
                            input_embedding[x].append(
                                torch.zeros(len(token[x]), device=local_rank).unsqueeze(
                                    0
                                )
                            )
                            attention_mask[x].append(torch.zeros(1))
                        else:
                            # input_embedding[x].append(token[x].unsqueeze(0).to(self.device))
                            # attention_mask[x].append(torch.ones(1, device=self.device))
                            input_embedding[x].append(token[x].unsqueeze(0))
                            attention_mask[x].append(torch.ones(1))
            except:
                print(index)
        for x in range(len(input_embedding)):
            input_embedding[x] = torch.concatenate(input_embedding[x], dim=0).unsqueeze(
                0
            )
            attention_mask[x] = torch.concatenate(attention_mask[x], dim=0).unsqueeze(0)
        input_embedding = torch.concatenate(input_embedding, dim=0).to(torch.float32)
        attention_mask = (
            torch.concatenate(attention_mask, dim=0).to(torch.float32).to(local_rank)
        )

        return input_embedding, attention_mask


class Scaler:
    def __init__(self, max_vector, min_vector):
        super(Scaler, self).__init__()
        self.max_vector = max_vector
        self.min_vector = min_vector

    def normalize(self, vector):
        vector = np.array(vector)
        normalized_vector = (vector - self.min_vector) / (
            self.max_vector - self.min_vector + 1e-8
        )
        vector = normalized_vector.tolist()
        return vector


class MainDataset(Dataset):
    def __init__(
        self,
        path,
        datasets,
        mode,
        model_name,
        hidden_dim,
        tokenizer,
        scaler=None,
        intention_scaler=None,
        generate_answer=False,
    ):
        self.path = path
        self.trajs = []
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.mode = mode
        min_vectors = np.full(hidden_dim, 1e8)
        min_int_vectors = np.full(hidden_dim, 1e8)
        max_vectors = np.full(hidden_dim, -1e8)
        max_int_vectors = np.full(hidden_dim, -1e8)
        if generate_answer:
            self.main_dataset = []
            for dataset in datasets:
                city, stage = dataset.split("-")
                with open(
                    f"{path}/util_datas/main_dataset/{city}/{stage}.json", "r"
                ) as f:
                    data = json.load(f)
                self.main_dataset.extend(data)
        file_list = os.listdir(f"{path}/util_datas/traj_dataset/{model_name}/{mode}")
        for file in tqdm(random.sample(file_list, int(1 * len(file_list)))):
            # change to full when run
            with open(
                os.path.join(
                    f"{path}/util_datas/traj_dataset/{model_name}/{mode}", file
                ),
                "r",
            ) as f:
                traj = json.load(f)
            ret_next_intention = [traj["next_intention"]]
            ret_disaster_level = [traj["disaster_level"]]
            token_types = traj["token_type"]
            tokens = []
            masks = []
            max_len = 50
            for index, x in enumerate(traj["tokens"]):
                if token_types[index] == 0:
                    tokens.append(x)
                    masks.append(True)
                    end_index = index + max_len
                else:
                    if token_types[index] == 2:
                        tokens.append(x)
                        masks.append(False)
                        min_int_vectors = np.min(
                            np.column_stack((min_int_vectors, x)), axis=1
                        )
                        max_int_vectors = np.max(
                            np.column_stack((max_int_vectors, x)), axis=1
                        )
                    elif index < end_index:
                        tokens.append(x)
                        masks.append(False)
                        min_vectors = np.min(np.column_stack((min_vectors, x)), axis=1)
                        max_vectors = np.max(np.column_stack((max_vectors, x)), axis=1)
            tokens = [x[0] if len(x) == 1 else x for x in tokens]
            self.trajs.append(
                {
                    "next_intention": ret_next_intention,
                    "disaster_level": ret_disaster_level,
                    "tokens": tokens,
                    "masks": masks,
                    "token_types": token_types,
                    "change_or_not": traj["change_or_not"],
                    "change_to_immob": traj["change_to_immob"],
                }
            )
        if scaler == None:
            self.scaler = Scaler(min_vector=min_vectors, max_vector=max_vectors)
        else:
            self.scaler = scaler
        if intention_scaler == None:
            self.intention_scaler = Scaler(
                min_vector=min_int_vectors, max_vector=max_int_vectors
            )
        else:
            self.intention_scaler = intention_scaler
        with open(f"{path}/util_datas/intention_emb/{model_name}.json", "r") as f:
            self.intention_emb = json.load(f)
        self.intention_emb[str(len(self.intention_emb.keys()))] = np.zeros(
            len(self.intention_emb["0"])
        ).tolist()

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        traj = self.trajs[idx]
        next_intention = traj["next_intention"]
        disaster_level = traj["disaster_level"]
        masks = traj["masks"]
        token_types = traj["token_types"]
        tokens = []
        for index, x in enumerate(traj["tokens"]):
            if masks[index]:
                tokens.append(x)
            else:
                if token_types[index] == 2:
                    tokens.append(self.intention_scaler.normalize(x))
                else:
                    tokens.append(self.scaler.normalize(x))
        change_or_not = traj["change_or_not"]
        change_to_immob = traj["change_to_immob"]
        labels = []
        if change_or_not == 1:
            labels.append("yes")
            if change_to_immob == 1:
                labels.append("yes")
            else:
                labels.append("no")
            labels.append(str(next_intention[0]))
        else:
            labels = ["no", "no", str(next_intention[0])]
        label_str = ",".join(labels)
        label = f"[{label_str}]"
        label = self.tokenizer(
            label, max_length=8, padding="max_length", return_tensors="pt"
        )
        if self.mode == "test":
            labels = [-100 for _ in range(20)]
            labels_marker = [False for _ in range(20)]
            for index, token in enumerate(tokens):
                if masks[index]:
                    if not index == len(tokens) - 1:
                        for t in token:
                            labels.append(-100)
                            labels_marker.append(False)
                    else:
                        for t in token:
                            labels.append(t)
                            labels_marker.append(True)
                else:
                    labels.append(-100)
                    labels_marker.append(False)
            pred_label = label.data["input_ids"].numpy().tolist()[0][1:]
            return (
                next_intention,
                disaster_level,
                tokens,
                masks,
                labels,
                labels_marker,
                pred_label,
            )
        else:
            tokens.append(label.data["input_ids"].numpy().tolist()[0][1:])
            masks.append(True)
            labels = [-100 for _ in range(20)]
            labels_marker = [False for _ in range(20)]
            for index, token in enumerate(tokens):
                if masks[index]:
                    if not index == len(tokens) - 1:
                        for t in token:
                            labels.append(-100)
                            labels_marker.append(False)
                    else:
                        for t in token:
                            labels.append(t)
                            labels_marker.append(True)
                else:
                    labels.append(-100)
                    labels_marker.append(False)
            return (
                next_intention,
                disaster_level,
                tokens,
                masks,
                labels,
                labels_marker,
                1,
            )

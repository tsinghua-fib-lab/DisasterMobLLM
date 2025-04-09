import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import json
import torch
from tqdm import *
from CLIP_model import *
from utils import *
from torch.nn.functional import normalize


class IntentionEmbedding:
    def __init__(self, trajectory_tokenizor, trajectory_encoder):
        self._model = trajectory_encoder
        self._tokenizer = trajectory_tokenizor

    def get_embedding(self, traj):
        intention_emb_list = []
        # traj = [[traj[j] for j in range(i + 1)] for i in range(len(traj))]
        for point in traj["st_emb_seq"]:
            # encoded_input = self._tokenizer(point)
            # encoded_input = np.expand_dims(encoded_input, axis=0)
            encoded_input = np.expand_dims(point, axis=0)
            encoded_input = np.expand_dims(encoded_input, axis=0)
            encoded_input = torch.tensor(encoded_input, dtype=torch.float32).cuda()
            with torch.no_grad():
                model_output, _ = self._model(encoded_input)
            intention_emb_list.append(model_output.cpu().numpy().tolist())
        return intention_emb_list

    def get_embedding_retrieval(self, traj):
        intention_emb_list = []
        # traj = [[traj[j] for j in range(i + 1)] for i in range(len(traj))]
        for point in traj["st_emb_seq"]:
            if not point[64] == 0:
                encoded_input = np.expand_dims(point, axis=0)
                encoded_input = np.expand_dims(encoded_input, axis=0)
                encoded_input = torch.tensor(encoded_input, dtype=torch.float32).cuda()
                with torch.no_grad():
                    model_output, logits = self._model(encoded_input)
                intention_emb_list.append(model_output.cpu().numpy().tolist()[0])
            else:
                intention_emb_list.append(np.zeros(self._model.hidden_dim).tolist())
        return intention_emb_list

    @classmethod
    def DTW(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        n, m = len(vector1), len(vector2)
        dtw_matrix = np.zeros((n + 1, m + 1))

        # 初始化边界
        for i in range(1, n + 1):
            dtw_matrix[i, 0] = float("inf")
        for j in range(1, m + 1):
            dtw_matrix[0, j] = float("inf")
        dtw_matrix[0, 0] = 0

        # 动态规划填充矩阵
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = euclidean_distance(vector1[i - 1], vector2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
                )

        return dtw_matrix[n, m]


class ReadFiles:
    """
    class to read files
    """

    def __init__(self, path, dataset):
        self._path = path
        self.datasets = dataset
        self.file_list = self.get_files()

    def get_files(self):
        # args：dir_path，目标文件夹路径
        file_list = []
        for dataset in self.datasets:
            city, mode = dataset.split("-")
            fpath = f"{self._path}/util_datas/main_dataset/{city}/{mode}.json"
            file_list.append(fpath)
        return file_list

    def get_content(self):
        docs = []
        # 读取文件内容
        for file in self.file_list:
            content = self.read_file_content(file)
            docs.extend(content)
        return docs

    @classmethod
    def read_file_content(cls, file_path: str):
        # 根据文件扩展名选择读取方法
        if file_path.endswith(".pk"):
            return cls.read_trajs(file_path, "pk")
        if file_path.endswith(".json"):
            return cls.read_trajs(file_path, "json")
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def read_trajs(cls, file_path, type):
        if type == "pk":
            with open(file_path, "rb") as file:
                reader = pickle.load(file)
                trajs = []
                for user in reader["data_filter"]:
                    user_data = reader["data_filter"][user]["sessions"]
                    for sid in user_data:
                        traj = []
                        for point in user_data[sid]:
                            traj.append(point[0])
                        trajs.append(traj)
                return trajs
        if type == "json":
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    data = json.load(file)
                    return data
            else:
                return []
        else:
            print("wrong data type")
            exit(0)


class Documents:
    """
    获取已分好类的json格式文档
    """

    def __init__(self, path: str = "") -> None:
        self.path = path

    def get_content(self):
        with open(self.path, mode="r", encoding="utf-8") as f:
            content = json.load(f)
        return content


class VectorStore:
    def __init__(self, document: List[str] = [""]) -> None:
        self.document = document
        if not len(document) == 1:
            self.disaster_level = [x["disaster_level"] for x in self.document]

    def get_vector(self, EmbeddingModel: IntentionEmbedding):

        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def persist(self, path: str = "storage"):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/doecment.json", "w", encoding="utf-8") as f:
            json.dump(self.document, f, ensure_ascii=False)
        if self.vectors:
            with open(f"{path}/vectors.json", "w", encoding="utf-8") as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = "storage"):
        with open(f"{path}/vectors.json", "r", encoding="utf-8") as f:
            vectors = json.load(f)
            self.vectors = []
            for vec in vectors:
                self.vectors.append(vec)
        with open(f"{path}/doecment.json", "r", encoding="utf-8") as f:
            self.document = json.load(f)
        self.disaster_level = [x["disaster_level"] for x in self.document]

    def get_similarity_DTW(self, vector1: List[float], vector2: List[float]) -> float:
        return IntentionEmbedding.DTW(vector1, vector2)

    def get_similarity(
        self,
        query_vector,
        vectors,
        method="cos",
    ):
        if method == "cos":
            max_len = max([len(x) for x in vectors])
            pad_query_vector = np.pad(
                np.array(query_vector),
                ((0, max_len - len(query_vector)), (0, 0)),
                mode="constant",
                constant_values=0,
            )

            pad_vectors = np.concatenate(
                [
                    np.expand_dims(
                        np.pad(
                            np.squeeze(np.array(x), axis=1),
                            ((0, max_len - len(x)), (0, 0)),
                            mode="constant",
                            constant_values=0,
                        ),
                        axis=0,
                    )
                    for x in vectors
                ],
                axis=0,
            )

            pad_query_vector = normalize(
                torch.tensor(pad_query_vector, dtype=torch.float32).cuda(),
                dim=1,
            )
            pad_vectors = normalize(
                torch.tensor(pad_vectors, dtype=torch.float32).cuda(),
                dim=2,
            )
            cos_similarities = torch.bmm(
                pad_vectors,
                pad_query_vector.unsqueeze(0)
                .expand(pad_vectors.size(0), -1, -1)
                .transpose(1, 2),
            ).sum(dim=(1, 2))
            result = cos_similarities.detach().cpu().numpy()
        elif method == "DTW":
            result = np.array(
                [self.get_similarity_DTW(query_vector, vector) for vector in vectors]
            )
        return result

    def query(
        self,
        query,
        disaster_level,
        EmbeddingModel: IntentionEmbedding,
        k: int = 1,
        return_sim=False,
    ):
        query_vector = EmbeddingModel.get_embedding_retrieval(query)
        if disaster_level in self.disaster_level:
            dis_index = np.where(np.array(self.disaster_level) == disaster_level)[0]
            vectors = [self.vectors[x] for x in dis_index]
        else:
            vectors = self.vectors
        result = self.get_similarity(query_vector, vectors, method="cos")
        indices = result.argsort()[-k:][::-1]
        output = []
        for index in indices:
            if disaster_level in self.disaster_level:
                output.append(self.document[dis_index[index]])
            else:
                output.append(self.document[index])
        if return_sim:
            sims = result[indices].tolist()
            return output, sims
        else:
            return output

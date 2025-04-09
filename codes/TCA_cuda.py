import numpy as np
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, FunctionTransformer

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
city_datas = []
data_len = []

for dataset in datasets:
    vectors = []
    for mode in modes:
        dataset_vecs = []
        with open(
            f"/your_path/Mob_data/datas/util_datas/CLIP_dataset/{dataset}/{mode}.json",
            "r",
        ) as f:
            data = json.load(f)
        for session in data:
            for index in range(len(session["vectors"])):
                # vector = (
                #     session["vectors"][index]
                #     + np.array(session["time_emb"][index])[
                #         [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 16]
                #     ].tolist()
                # )
                vector = np.array(session["vectors"][index])[:66].tolist()

                # vector = session["vectors"][index] + session["time_emb"][index]
                if not vector[64] == 0:
                    dataset_vecs.append(vector)
        dataset_vecs = np.array(dataset_vecs)
        epsilon = 1e-8
        dataset_vecs = dataset_vecs + epsilon
        scaler = MinMaxScaler()
        dataset_vecs = scaler.fit_transform(dataset_vecs)
        # l1_normalizer = Normalizer(norm="l2")
        # dataset_vecs = l1_normalizer.fit_transform(dataset_vecs)
        # log_transformer = FunctionTransformer(np.log1p, validate=True)
        # dataset_vecs = log_transformer.transform(dataset_vecs)
        # box_cox_transformer = PowerTransformer(method='box-cox')
        # dataset_vecs = box_cox_transformer.fit_transform(dataset_vecs)  # 确保数据非负
        vectors.extend(dataset_vecs.tolist())
    data_len.append(len(vectors))
    city_datas.extend(np.array(vectors))

vectors = np.array(city_datas)
print(vectors.shape)


def perform_pca(data, n_components):
    pca = PCA(n_components=n_components)
    low_dim_data = pca.fit_transform(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    print("Explained Variance Ratio:", explained_variance_ratio)
    print("Cumulative Explained Variance:", cumulative_explained_variance)

    plt.figure(figsize=(8, 6))
    plt.plot(
        range(1, len(cumulative_explained_variance) + 1),
        cumulative_explained_variance,
        marker="o",
    )
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance by Number of Components")
    plt.grid(True)
    plt.show()

    return low_dim_data


# region_emb = np.concatenate(
#     (
#         np.concatenate((vectors[:, 0:22], vectors[:, 22:44]), axis=0),
#         np.concatenate((vectors[:, 44:49], vectors[:, 49:54]), axis=0),
#         np.concatenate((vectors[:, 54:59], vectors[:, 59:64]), axis=0),
#     ),
#     axis=1,
# )

# n_components_region = 16
# region_emb_low = perform_pca(region_emb, n_components_region)

# relation_emb = vectors[:, 64:94]
# n_components_relation = 15
# relation_emb_low = perform_pca(relation_emb, n_components_relation)

# time_emb = vectors[:, 94:]
# n_components_time = 12
# time_emb_low = perform_pca(time_emb, n_components_time)

# print(region_emb_low.shape, relation_emb_low.shape, time_emb_low.shape)

# new_dataset = np.concatenate(
#     (
#         region_emb_low[int(region_emb_low.shape[0] / 2) :, :],
#         region_emb_low[: int(region_emb_low.shape[0] / 2), :],
#         relation_emb_low,
#     ),
#     axis=1,
# )


def rbf_kernel(X, Y=None, gamma=1.0):
    if Y is None:
        Y = X
    K = -2 * torch.mm(X, Y.t())
    K += torch.sum(X**2, dim=1, keepdim=True)
    K += torch.sum(Y**2, dim=1, keepdim=True).t()
    K *= -gamma
    return torch.exp(K)


def multi_domain_tca(Xs, dim=5, kernel_type="rbf", gamma=1.0, mu=1.0):
    # 将所有领域的数据堆叠在一起
    X = torch.vstack(Xs)
    total_samples = sum(X.shape[0] for X in Xs)
    ns = [X.shape[0] for X in Xs]

    print("Calculating kernel matrix...")
    if kernel_type == "rbf":
        K = rbf_kernel(X, gamma=gamma)
    elif kernel_type == "linear":
        K = torch.mm(X, X.t())
    else:
        raise ValueError("Unsupported kernel type: {}".format(kernel_type))

    print("Building cross-domain similarity matrix...")
    L = torch.zeros((total_samples, total_samples)).to(device)
    start_idx = 0
    for n in tqdm(ns):
        end_idx = start_idx + n
        L[start_idx:end_idx, start_idx:end_idx] = 1.0 / n * torch.ones((n, n))
        start_idx = end_idx

    I = torch.eye(total_samples)
    H = I - 1.0 / total_samples * torch.ones((total_samples, total_samples))
    H = H.to(device)
    M = torch.mm(torch.mm(H, K), H) + mu * torch.mm(L, K)

    print("Solving eigenvalue problem...")
    eigvals, eigvecs = torch.linalg.eigh(M)
    # 取最大的dim个特征向量
    eigvals = eigvals[-dim:]
    eigvecs = eigvecs[:, -dim:]
    V = eigvecs / torch.sqrt(torch.abs(eigvals))

    print("Applying transformation matrix...")
    Xs_new = [
        torch.mm(K[start_idx : start_idx + n, :], V)
        for start_idx, n in zip(np.cumsum([0] + ns[:-1]), ns)
    ]

    return Xs_new


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

# 将数据转换为PyTorch张量并移动到GPU
vectors_tensor = torch.tensor(vectors, dtype=torch.float32).to(device)
city_datas = []
start = 0
for l in data_len:
    city_datas.append(vectors_tensor[start : start + l, :])
    start = start + l

Xs = city_datas
print("Starting TCA...")

# 使用多领域TCA进行领域适应
Xs_new = multi_domain_tca(Xs, dim=20, kernel_type="rbf", gamma=1.0, mu=1.0)

# 将结果转换回NumPy数组
Xs_new_np = [x.cpu().numpy() for x in Xs_new]

# 打印结果
for i, X_new in enumerate(Xs_new_np):
    print(f"Domain {i+1} transformed shape: {X_new.shape}")
    with open(
        f"/your_path/Mob_data/datas/util_datas/CLIP_input/{datasets[i]}.json",
        "w",
    ) as f:
        json.dump(X_new.tolist(), f)

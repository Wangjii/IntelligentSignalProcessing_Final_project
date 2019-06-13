from load_data import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


def PCA_process(data_addr):
    pca = PCA(n_components=0.85)
    data_addr_pca = pca.fit_transform(data_addr)
    return data_addr_pca


def PCA_process_test(data_addr):
    # PCA
    pca = PCA(n_components=13)
    data_pca = pca.fit(data_addr)
    # 各特征百分比
    np.set_printoptions(precision=5, suppress=True)
    print("各个特征的方差百分比为:\n", data_pca.explained_variance_ratio_)
    print("各个特征的方差为:\n", data_pca.explained_variance_)

    pca = PCA(n_components=0.85)
    data_addr_pca = pca.fit_transform(data_addr)
    print("占总方差85%所需的维数为{}".format(pca.n_components_))
    return data_addr_pca


def main():
    file_path = "wine.csv"
    data_class, data_addr = load_data(file_path)
    # 数据预处理 归一化
    data_addr = StandardScaler().fit_transform(data_addr)
    # PCA 处理
    PCA_process_test(data_addr)


if __name__ == "__main__":
    main()

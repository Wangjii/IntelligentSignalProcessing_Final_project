from load_data import load_data
from draw import draw_single
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA


def KPCA_process(data_addr):
    kpca = KernelPCA(kernel="rbf", n_components=2)
    data_addr_kpca = kpca.fit_transform(data_addr)
    return data_addr_kpca


def main():
    file_path = "wine.csv"
    data_class, data_addr = load_data(file_path)
    # 数据预处理 归一化
    data_addr = StandardScaler().fit_transform(data_addr)
    data_addr_kpca = KPCA_process(data_addr)
    draw_single(data_class, data_addr_kpca, 'KPCA data')


if __name__ == "__main__":
    main()

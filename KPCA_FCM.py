# KPCA and FCM
from load_data import load_data
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans


def CMEANS_process(data_addr_pca):
    train = data_addr_pca.T
    center, u, u0, d, jm, p, fpc = cmeans(train,
                                          m=2,
                                          c=3,
                                          error=0.0001,
                                          maxiter=100)
    class_1 = 0
    class_2 = 0
    class_3 = 0
    for i in u:
        label = np.argmax(u, axis=0)
    print(label)
    for i in range(178):
        if label[i] == 0:
            class_1 += 1
        elif label[i] == 1:
            class_2 += 1
        elif label[i] == 2:
            class_3 += 1
    print(class_1, class_2, class_3)

    plt.figure()
    ax1 = plt.subplot(121)
    ax1.set_title('Classify data')
    for i in range(178):
        if label[i] == 0:
            plt.scatter(train[0][i], train[1][i], c='r')
        elif label[i] == 1:
            plt.scatter(train[0][i], train[1][i], c='g')
        elif label[i] == 2:
            plt.scatter(train[0][i], train[1][i], c='b')

    ax2 = plt.subplot(122)
    ax2.set_title('Original data')
    for i in range(178):
        if data_class[i] - 1 == 0:
            plt.scatter(train[0][i], train[1][i], c='r')
        elif data_class[i] - 1 == 1:
            plt.scatter(train[0][i], train[1][i], c='g')
        elif data_class[i] - 1 == 2:
            plt.scatter(train[0][i], train[1][i], c='b')
    plt.show()


if __name__ == "__main__":
    file_path = "wine.csv"
    data_class, data_addr = load_data(file_path)
    # 数据预处理 归一化
    data_addr = StandardScaler().fit_transform(data_addr)
    # KPCA
    kpca = KernelPCA(kernel="rbf", n_components=2)
    data_addr_kpca = kpca.fit_transform(data_addr)
    # FCM 处理
    CMEANS_process(data_addr_kpca)

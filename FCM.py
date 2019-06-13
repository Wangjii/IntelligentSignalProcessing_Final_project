import numpy as np
from load_data import load_data
from sklearn.preprocessing import StandardScaler
from LDA import LDA_process
from skfuzzy.cluster import cmeans


def CMEANS_process(data_addr_pca):
    train = data_addr_pca.T
    center, u, u0, d, jm, p, fpc = cmeans(train,
                                          m=2,
                                          c=3,
                                          error=0.0001,
                                          maxiter=100)

    for i in u:
        label = np.argmax(u, axis=0)

    class_1 = 0
    class_2 = 0
    class_3 = 0

    for i in range(178):
        if label[i] == 0:
            class_1 += 1
        elif label[i] == 1:
            class_2 += 1
        elif label[i] == 2:
            class_3 += 1
    print('第一类有{:};第二类有{:};第三类有{:}'.format(class_1, class_2, class_3))

    return label + 1


if __name__ == "__main__":
    file_path = "wine.csv"
    data_class, data_addr = load_data(file_path)
    data_addr = StandardScaler().fit_transform(data_addr)
    # LDA
    data_addr_lda = LDA_process(data_class, data_addr)
    # FCM 处理
    label = CMEANS_process(data_addr_lda)

    target = label == data_class
    error = 0
    for mark in target:
        if mark == False:
            error = error + 1

    print("分类错误的个数为{}".format(error))

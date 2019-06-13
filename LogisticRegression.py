# LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from load_data import load_data
from LDA import LDA_process
import matplotlib.pyplot as plt
import numpy as np


def draw_double(data_class1, data_addr1, str1, data_class2, data_addr2, str2,
                Lr):
    plt.figure()
    ax1 = plt.subplot(121)
    ax1.set_title(str1)
    x_min = -6
    x_max = 6
    y_min = -5
    y_max = 5
    for i in range(len(data_addr1)):
        if data_class1[i] == 1:
            plt.scatter(data_addr1[i][0], data_addr1[i][1], c='r')
        elif data_class1[i] == 2:
            plt.scatter(data_addr1[i][0], data_addr1[i][1], c='g')
        elif data_class1[i] == 3:
            plt.scatter(data_addr1[i][0], data_addr1[i][1], c='b')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    ax2 = plt.subplot(122)
    ax2.set_title(str2)

    XX, YY = np.mgrid[x_min:x_max:300j, y_min:y_max:300j]
    Z = Lr.predict(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

    for i in range(len(data_addr2)):
        if data_class2[i] == 1:
            plt.scatter(data_addr2[i][0], data_addr2[i][1], c='r')
        elif data_class2[i] == 2:
            plt.scatter(data_addr2[i][0], data_addr2[i][1], c='g')
        elif data_class2[i] == 3:
            plt.scatter(data_addr2[i][0], data_addr2[i][1], c='b')
    plt.show()


def main():
    file_path = "wine.csv"
    data_class, data_addr = load_data(file_path)
    data_addr = StandardScaler().fit_transform(data_addr)
    data_addr_lda = LDA_process(data_class, data_addr)

    class_train, class_test, addr_train, addr_test = train_test_split(
        data_class, data_addr_lda, test_size=0.30, random_state=25)

    Lr = LogisticRegression()
    Lr.fit(addr_train, class_train)
    class_predict = Lr.predict(addr_test)
    draw_double(class_test, addr_test, 'Test data', class_predict, addr_test,
                'Predict data', Lr)

    print("平均正确率为{}".format(Lr.score(addr_test, class_test)))

    target = class_test == class_predict
    error = 0
    for mark in target:
        if (not mark):
            error = error + 1

    print("分类错误的个数为{}".format(error))


if __name__ == "__main__":
    main()

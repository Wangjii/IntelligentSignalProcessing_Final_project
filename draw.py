import matplotlib.pyplot as plt


def draw_single(data_class, data_addr, str):
    plt.figure()
    ax = plt.subplot(111)
    ax.set_title(str)
    for i in range(len(data_addr)):
        if data_class[i] == 1:
            plt.scatter(data_addr[i][0], data_addr[i][1], c='r')
        elif data_class[i] == 2:
            plt.scatter(data_addr[i][0], data_addr[i][1], c='g')
        elif data_class[i] == 3:
            plt.scatter(data_addr[i][0], data_addr[i][1], c='b')
    plt.show()


def draw_double(data_class1, data_addr1, str1, data_class2, data_addr2, str2):
    plt.figure()
    ax1 = plt.subplot(121)
    ax1.set_title(str1)
    for i in range(len(data_addr1)):
        if data_class1[i] == 1:
            plt.scatter(data_addr1[i][0], data_addr1[i][1], c='r')
        elif data_class1[i] == 2:
            plt.scatter(data_addr1[i][0], data_addr1[i][1], c='g')
        elif data_class1[i] == 3:
            plt.scatter(data_addr1[i][0], data_addr1[i][1], c='b')

    ax2 = plt.subplot(122)
    ax2.set_title(str2)
    for i in range(len(data_addr2)):
        if data_class2[i] == 1:
            plt.scatter(data_addr2[i][0], data_addr2[i][1], c='r')
        elif data_class2[i] == 2:
            plt.scatter(data_addr2[i][0], data_addr2[i][1], c='g')
        elif data_class2[i] == 3:
            plt.scatter(data_addr2[i][0], data_addr2[i][1], c='b')
    plt.show()

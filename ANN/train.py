# coding=utf-8

from model import neuralNetwork
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# number of input, hidden and output nodes
input_nodes = 13
hidden1_nodes = 20
hidden2_nodes = 15
output_nodes = 3

# Learning rate is 0.1
learning_rate = 0.1

RANDOM_STATE = 40


# 获取数据集
def load_data(file_path):
    f = open(file_path, 'rt')
    data = []
    for line in f:
        line = line.replace("\n", "")
        data.append(eval(line))
    data = numpy.array(data)
    data_class = data[:, 0]
    data_addr = data[:, 1:]

    return data_class, data_addr


# create instance of neural network
n = neuralNetwork(input_nodes, hidden1_nodes, hidden1_nodes, output_nodes,
                  learning_rate)

# 数据路径
file_path = "wine.csv"
# 读入数据
data_class, data_addr = load_data(file_path)

targets = numpy.zeros((len(data_class), 3)) + 0.01
for e in range(len(data_class)):
    if data_class[e] == 1:
        targets[e, 0] = 0.99
    elif data_class[e] == 2:
        targets[e, 1] = 0.99
    elif data_class[e] == 3:
        targets[e, 2] = 0.99

data_addr = StandardScaler().fit_transform(data_addr)

# divide into train data and test data
addr_train, addr_test, class_train, class_test = train_test_split(
    data_addr, targets, test_size=0.30, random_state=RANDOM_STATE)

# 对训练过程进行循环
epochs = 20
for e in range(epochs):
    for record in range(len(addr_train)):
        inputs = addr_train[record, :]
        target = class_train[record, :]
        # 传入网络进行训练
        n.train(inputs, target)
        pass
    pass

# 创建一个空白的计分卡
scorecard = []
# 遍历测试数据
for record in range(len(addr_test)):
    inputs = addr_test[record, :]
    # 通过神经网络得出结果
    outputs = n.query(inputs)
    # 结果
    label = numpy.argmax(outputs)
    # print(label, "network's answer")
    # 分类相同，计分卡加一，否则加零
    if (label == numpy.argmax(class_test[record, :])):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass
# 输出计分卡
print(scorecard)
# 输出分数
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)

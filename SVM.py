# SVM
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from load_data import load_data
from LDA import LDA_process
from LogisticRegression import draw_double


def main():
    file_path = 'wine.csv'
    data_class, data_addr = load_data(file_path)
    data_addr = StandardScaler().fit_transform(data_addr)
    data_addr = LDA_process(data_class, data_addr)
    addr_train, addr_test, class_train, class_test = train_test_split(
        data_addr, data_class, test_size=0.30, random_state=30)
    clf = SVC(kernel='rbf', gamma=1)
    clf.fit(addr_train, class_train)
    class_predict = clf.predict(addr_test)

    target = class_test == class_predict
    error = 0
    for mark in target:
        if (not mark):
            error = error + 1

    print("分类错误的个数为{}".format(error))

    draw_double(class_test, addr_test, 'Original Test data', class_predict,
                addr_test, 'Predict Test data', clf)


if __name__ == "__main__":
    main()

from load_data import load_data
from draw import draw_double
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def main():
    file_path = "wine.csv"  # input data-file's path
    data_class, data_addr = load_data(file_path)  # load data-sheet
    data_addr = StandardScaler().fit_transform(data_addr)
    addr_train, addr_test, class_train, class_test = train_test_split(
        data_addr, data_class, test_size=0.30,
        random_state=30)  # divide into train data and test data

    lda = LinearDiscriminantAnalysis(n_components=2)  # LDA
    lda.fit(addr_train, class_train)
    addr_test_lda = lda.transform(addr_test)
    class_pred = lda.predict(addr_test)
    draw_double(class_test, addr_test_lda, 'Original data', class_pred,
                addr_test_lda, 'LDA data')

    target = class_test == class_pred
    error = 0
    for mark in target:
        if (not mark):
            error = error + 1

    print("分类错误的个数为{}".format(error))


if __name__ == "__main__":
    main()

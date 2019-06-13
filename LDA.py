from load_data import load_data
from draw import draw_single
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def LDA_process(data_class, data_addr):
    lda = LinearDiscriminantAnalysis(n_components=2)  # LDA
    data_addr_lda = lda.fit_transform(data_addr, data_class)
    return data_addr_lda


def main():
    file_path = "wine.csv"  # input data-file's path
    data_class, data_addr = load_data(file_path)  # load data-sheet
    data_addr = StandardScaler().fit_transform(data_addr)
    data_addr_lda = LDA_process(data_class, data_addr)
    draw_single(data_class, data_addr_lda, 'LDA data')


if __name__ == "__main__":
    main()

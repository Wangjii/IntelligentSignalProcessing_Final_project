from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from load_data import load_data

FIG_SIZE = (10, 7)


def draw_figure(data_1, data_2, data_class):
    # 作图
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)
    ax1.set_title('Original data')
    ax2.set_title('Standard data')

    for l, c, m in zip(range(1, 4), ('blue', 'red', 'green'), ('^', 's', 'o')):
        ax1.scatter(data_1[data_class == l, 0],
                    data_1[data_class == l, 1],
                    color=c,
                    label='class %s' % l,
                    alpha=0.5,
                    marker=m)

    for l, c, m in zip(range(1, 4), ('blue', 'red', 'green'), ('^', 's', 'o')):
        ax2.scatter(data_2[data_class == l, 0],
                    data_2[data_class == l, 1],
                    color=c,
                    label='class %s' % l,
                    alpha=0.5,
                    marker=m)

    for ax in (ax1, ax2):
        ax.grid()

    plt.show()


def main():
    file_path = "wine.csv"  # 数据路径
    data_class, data_addr = load_data(file_path)  # 读入数据
    pca = PCA(n_components=2)  # PCA处理
    data_addr_pca = pca.fit_transform(data_addr)  # 原始数据PCA处理
    data_addr_std = StandardScaler().fit_transform(data_addr)  # 标准化处理
    data_addr_std_pca = pca.fit_transform(data_addr_std)  # 标准化处理数据PCA
    draw_figure(data_addr_pca, data_addr_std_pca, data_class)  # 可视化作图


if __name__ == "__main__":
    main()

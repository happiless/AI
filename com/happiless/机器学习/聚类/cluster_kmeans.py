import sklearn.datasets as ds
import matplotlib.pyplot as plt


def main():
    N = 400
    centers = 4
    data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=2)
    print(data, y)
    plt.plot(data, y)
    plt.show()


if __name__ == '__main__':
    main()
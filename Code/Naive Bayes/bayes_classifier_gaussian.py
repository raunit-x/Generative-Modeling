import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal as mvn


class BayesClassifier:
    def __init__(self):
        self.K = 0
        self.gaussian_variables = []

    def fit(self, X, Y):
        self.K = len(set(Y))
        for k in range(self.K):
            Xk = X[Y == k]
            self.gaussian_variables.append({'m': Xk.mean(axis=0), 'c': np.cov(Xk.T)})

    def sample_given_y(self, y):
        g = self.gaussian_variables[y]
        return mvn.rvs(mean=g['m'], cov=g['c'])

    def sample(self):
        y = np.random.randint(self.K)
        return self.sample_given_y(y)


def read_data():
    df = pd.read_csv('/Users/raunit_x/Desktop/Dataset/Digit Recognizer/train.csv')
    Y = df[df.columns[0]].to_numpy()
    X = df[df.columns[1:]].to_numpy()
    print(f"X: {type(X)}, {X.shape}")
    print(f"Y: {type(Y)}, {Y.shape}")
    return X, Y


def main():
    X, Y = read_data()
    clf = BayesClassifier()
    clf.fit(X, Y)

    for k in range(clf.K):
        sample = clf.sample_given_y(k).reshape(28, 28)
        mean = clf.gaussian_variables[k]['m'].reshape(28, 28)

        plt.subplot(1, 2, 1)
        plt.imshow(sample, cmap='gray')
        plt.title('Sample')
        plt.subplot(1, 2, 2)
        plt.imshow(mean, cmap='gray')
        plt.title('Mean')
        plt.show()

    sample = clf.sample().reshape(28, 28)
    plt.imshow(sample, cmap='gray')
    plt.title("Random Sample")
    plt.show()


if __name__ == '__main__':
    main()

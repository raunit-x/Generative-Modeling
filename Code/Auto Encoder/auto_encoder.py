import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


class AutoEncoder:
    def __init__(self, D, M):
        # represent a batch of training data
        self.X = tf.compat.v1.placeholder(tf.float32, shape=(None, D))

        # input to hidden
        self.W = tf.Variable(tf.random.normal(shape=(D, M)) * 2 / np.sqrt(M))
        self.b = tf.Variable(np.zeros(M).astype(np.float32))

        # hidden to output
        self.V = tf.Variable(tf.random.normal(shape=(M, D)) * 2 / np.sqrt(D))
        self.c = tf.Variable(np.zeros(D).astype(np.float32))

        # construct the reconstruction
        self.Z = tf.nn.relu(tf.matmul(self.X, self.W) + self.b)
        logits = tf.matmul(self.Z, self.V) + self.c
        self.X_hat = tf.math.sigmoid(logits)

        # compute the cost
        self.cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X_hat, logits=logits))

        # make the trainer
        self.train_op = tf.compat.v1.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.cost)

        # set up the sessions and variables for later
        self.init_op = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.InteractiveSession()
        self.sess.run(self.init_op)

    def fit(self, X, epochs=10, batch_size=64):
        costs = []
        n_batches = len(X) // batch_size
        print(f"n_batches: {n_batches}")
        for i in range(epochs):
            print(f"epoch: {i}")
            np.random.shuffle(X)
            for j in range(n_batches):
                batch = X[j * batch_size: (j + 1) * batch_size]
                _, c = self.sess.run((self.train_op, self.cost), feed_dict={self.X: batch})
                c /= batch_size
                costs.append(c)
                if j % 100 == 0:
                    print(f"Iter: {j}, Cost: {c}")
        plt.plot(costs)
        plt.show()

    def predict(self, X):
        return self.sess.run(self.X_hat, feed_dict={self.X: X})


def read_data():
    df = pd.read_csv('/Users/raunit_x/Desktop/Dataset/Digit Recognizer/train.csv')
    Y = df[df.columns[0]].to_numpy()
    X = df[df.columns[1:]].to_numpy()
    print(f"X: {type(X)}, {X.shape}")
    print(f"Y: {type(Y)}, {Y.shape}")
    return X, Y


def main():
    tf.compat.v1.disable_eager_execution()
    X, Y = read_data()
    model = AutoEncoder(784, 300)
    model.fit(X)
    done = False
    while not done:
        i = np.random.choice(len(X))
        x = X[i]
        im = model.predict([x]).reshape(28, 28)
        print(x.reshape(28, 28))
        print(im)
        plt.subplot(1, 2, 1)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.imshow(im, cmap='gray')
        plt.title("Reconstruction")
        plt.show()

        ans = input("Generate another?[Y/N]: ")
        done = ans == 'n' or ans == 'N'


if __name__ == '__main__':
    main()

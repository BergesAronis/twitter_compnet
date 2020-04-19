import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class hypothesis():

    def __init__(self, positive=["1"], negative=["0"], label_column="out", theta=[]):

        self.positive = positive
        self.negative = negative
        self.theta = theta
        self.label_column=label_column

    def train(self, testing_set, alpha, iters):
        df = testing_set.copy()
        # process testing_set
        self.process_data(df)
        self.gradient_descent(alpha, iters)



    def predict(self, inputs):
        return self.sigmoid(inputs.dot(self.theta))

    def sigmoid(self, input):
        return 1/(1+np.exp(-input))

    def gradient_descent(self, alpha, iterations):

        # Iterate a certain amount of times
        for i in range(iterations):
            theta_copy = np.copy(self.theta)
            h = self.sigmoid((self.X).dot(theta_copy))
            m = len(self.X)
            for t in range(len(self.theta)):
                tempx = self.X[:, t].reshape(len(self.X),1)
                self.theta[t] = theta_copy[t] - alpha * ((1/m) * sum(np.multiply((h - self.Y), tempx)))


    def cost(self):
        m = len(self.X)

        # H(X * theta) <--- Sigmoid function
        h = self.sigmoid((self.X).dot(self.theta))

        # Cost function
        J = (1/m) * np.sum((np.multiply(-self.Y, np.log(h))) - np.multiply((1 - self.Y), np.log(1 - h)))

        return J

    def plot_data(self, plot):
        plt.figure(plot, figsize=(11,6))
        for i in range(6):
            plt.subplot(230 + (i + 1))
            plt.plot(self.X[:,i], self.Y, 'ro')
            plt.ylabel('Y')
            plt.xlabel('X' + str(i))
        plt.show()

    def process_data(self, df):

        # Convert y values to 1 or 0
        for output in self.positive:
            df.loc[df[self.label_column] == output, self.label_column] = 1
        for output in self.negative:
            df.loc[df[self.label_column] == output, self.label_column] = 0

        # Generate X, Y and theta matrices
        matrix = df.values
        self.X = matrix[:, 0:len(matrix[0]) - 1]
        self.Y = matrix[:, len(matrix[0]) - 1]
        self.Y = self.Y.reshape(len(self.X), 1)
        self.theta = np.zeros((1,len(self.X[0]))).transpose()

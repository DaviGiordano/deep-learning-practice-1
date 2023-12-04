#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a

        # Predict the class label for the input example x_i by finding the class
        # with the highest score. The score for each class is computed as the dot
        # product of the weights (self.W) and the input features (x_i).
        y_hat_i = np.argmax(np.dot(self.W, x_i))

        # Check if the predicted class label (y_hat_i) is different from the true
        # class label (y_i). If they are different, update the weights.
        if y_hat_i != y_i:
            # For the true class label (y_i), increase the weights by x_i. This makes
            # it more likely that a similar input will be classified as y_i in the future.
            self.W[y_i] += x_i

            # For the incorrectly predicted class label (y_hat_i), decrease the weights by x_i.
            # This makes it less likely that a similar input will be misclassified as y_hat_i in the future.
            self.W[y_hat_i] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b

        # Compute the raw scores for each class by multiplying the weight matrix (self.W)
        # with the input feature vector (x_i).
        scores = np.dot(self.W, x_i)

        # Apply the softmax function to the scores to convert them into probabilities.
        # Subtracting the max score from each score for numerical stability to prevent
        # overflow in the exponential calculation.
        exp_scores = np.exp(scores - np.max(scores))

        # Normalize the exponentiated scores to get probabilities. This is done by
        # dividing each exponentiated score by the sum of all exponentiated scores.
        probabilities = exp_scores / np.sum(exp_scores)

        # Create a one-hot encoded vector for the true class label (y_i). This vector
        # has zeros in all positions except for the position corresponding to the true
        # class, which is set to 1.
        y_one_hot = np.zeros(self.W.shape[0])
        y_one_hot[y_i] = 1

        # Compute the gradient of the loss with respect to the weights. The gradient
        # is calculated as the outer product of the difference between the predicted
        # probabilities and the one-hot encoded true label vector with the input
        # feature vector (x_i). This results in a matrix where each column represents
        # the gradient for each class.
        gradient = np.outer(probabilities - y_one_hot, x_i)

        # Update the weights by subtracting the product of the learning rate and the
        # gradient from the current weights. This step adjusts the weights in the
        # direction that reduces the loss for the current training example.
        self.W -= learning_rate * gradient


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        mu = 0.1
        sigma = 0.01

        self.W1 = np.random.normal(mu, sigma, (n_features, hidden_size))
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.normal(mu, sigma, (hidden_size, n_classes))
        self.b2 = np.zeros(n_classes)

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def softmax(Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward_pass(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)

        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)

        return A1, A2

    def predict(self, X):
        _, A2 = self.forward_pass(X)
        return np.argmax(A2, axis=1)

    def evaluate(self, X, y):
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    @staticmethod
    def cross_entropy_loss(y_hat, y):
        y_one_hot = np.zeros_like(y_hat)
        y_one_hot[np.arange(len(y)), y] = 1

        loss = -np.sum(y_one_hot * np.log(y_hat + 1e-9)) / len(y)
        return loss

    def backpropagation(self, X, y, A1, A2):
        dZ2 = A2
        dZ2[np.arange(len(y)), y] -= 1
        dZ2 /= len(y)

        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (A1 > 0)

        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0)

        return dW1, db1, dW2, db2

    def train_epoch(self, X, y, learning_rate=0.001, batch_size=1):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)  # Shuffle the indices to ensure random batches

        total_loss = 0

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # Forward pass
            A1, A2 = self.forward_pass(X_batch)

            # Compute loss
            loss = self.cross_entropy_loss(A2, y_batch)
            total_loss += loss * (end_idx - start_idx)

            # Backpropagation
            dW1, db1, dW2, db2 = self.backpropagation(X_batch, y_batch, A1, A2)

            # Update weights and biases
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

        average_loss = total_loss / n_samples
        return average_loss

    """
    def train_epoch(self, X, y, learning_rate=0.001):
        A1, A2 = self.forward_pass(X)
        loss = self.cross_entropy_loss(A2, y)

        dW1, db1, dW2, db2 = self.backpropagation(X, y, A1, A2)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        return loss
    """


def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []

    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )

        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
    ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()

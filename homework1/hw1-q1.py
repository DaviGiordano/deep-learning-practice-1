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


# Q1.2b
class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        mu, sigma = 0.1, 0.1  # Mean and standard deviation for weight initialization

        # Initialize weights and biases for the hidden layer
        self.W1 = np.random.normal(mu, sigma, (hidden_size, n_features))
        self.b1 = np.zeros(hidden_size)

        # Initialize weights and biases for the output layer
        self.W2 = np.random.normal(mu, sigma, (n_classes, hidden_size))
        self.b2 = np.zeros(n_classes)

    @staticmethod
    def relu(z):
        # ReLU activation function
        return np.maximum(0, z)

    @staticmethod
    def softmax(z):
        # Adjusted Softmax activation function for single processing
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

    def forward_pass(self, x_i):
        # Forward Pass function for single processing
        # Compute activations for the hidden layer
        z1_i = np.dot(self.W1, x_i) + self.b1
        a1_i = self.relu(z1_i)

        # Compute activations for the output layer
        z2_i = np.dot(self.W2, a1_i) + self.b2
        a2_i = self.softmax(z2_i)
        return a1_i, a2_i

    def forward_pass_batch(self, X):
        # Forward Pass function for batch processing
        # Compute activations for the hidden layer
        z1 = np.dot(X, self.W1.T) + self.b1
        a1 = self.relu(z1)

        # Compute activations for the output layer
        z2 = np.dot(a1, self.W2.T) + self.b2
        a2 = self.softmax(z2)
        return a1, a2

    def predict(self, X):
        # Predict class labels for a batch of inputs
        _, a2 = self.forward_pass_batch(X)
        return np.argmax(a2, axis=1)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def backward_pass(self, x_i, y_i, a1_i, a2_i, learning_rate):
        # Convert y to one-hot encoding
        y_one_hot = np.zeros_like(a2_i)
        y_one_hot[y_i] = 1

        # Compute gradients for the output layer
        dZ2 = a2_i - y_one_hot
        dW2 = np.outer(dZ2, a1_i)
        db2 = np.sum(dZ2)

        # Compute gradients for the hidden layer
        dA1 = np.dot(dZ2, self.W2)
        dZ1 = dA1 * (a1_i > 0)  # Derivative of ReLU
        dW1 = np.outer(dZ1, x_i)
        db1 = np.sum(dZ1)

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        # Compute and return the loss
        loss = -np.sum(y_one_hot * np.log(a2_i + 1e-8))
        return loss

    def train_epoch(self, X, y, learning_rate=0.001):
        # Shuffle the dataset
        num_examples = X.shape[0]
        indices = np.arange(num_examples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        total_loss = 0

        # Stochastic gradient descent by processing each example individually
        for x_i, y_i in zip(X, y):
            # Forward pass for a single example
            a1_i, a2_i = self.forward_pass(x_i)

            # Backward pass and update weights for a single example
            loss = self.backward_pass(x_i, y_i, a1_i, a2_i, learning_rate)
            total_loss += loss

        # Return average loss over the epoch
        return total_loss / num_examples


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


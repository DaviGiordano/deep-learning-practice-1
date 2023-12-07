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
        y_hat = np.argmax(self.W.dot(x_i))
        if y_hat != y_i:
            self.W[y_i, :] += x_i
            self.W[y_hat, :] -= x_i
        

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        label_scores = np.expand_dims(self.W.dot(x_i), axis=1)
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1

        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))

        self.W = self.W + learning_rate * (y_one_hot - label_probabilities).dot(np.expand_dims(x_i, axis = 1).T)


class MLP(object):

    def __init__(self, n_classes, n_features, hidden_size):

        # Initialize an MLP with a single hidden layer.
        self.mu_W = 0.1
        self.sigma_W = 0.01
        self.n_classes = n_classes
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.W1 = np.random.normal(self.mu_W, self.sigma_W, (hidden_size, n_features)) # (output x input)
        self.b1 = np.random.normal(self.mu_W, self.sigma_W, hidden_size) # (output,)

        self.W2 = np.random.normal(self.mu_W, self.sigma_W, (n_classes, hidden_size))
        self.b2 = np.random.normal(self.mu_W, self.sigma_W, n_classes)

    
    def relu(self, X):
        return np.maximum(0, X)
    
    def softmax(self, X):
        """
        Apply the softmax function to each row of the array.
        Use for array with (n_examples x n_classes)
        """
        e_x = np.exp(X - np.max(X, axis=1).reshape(-1, 1))
        return e_x / e_x.sum(axis=1).reshape(-1, 1)
    
    def assert_shape(self, arr, expected_shape):
        """
        Asserts that a numpy array has a specific shape.
        Args:
        arr (numpy.ndarray): A numpy array.
        expected_shape (tuple): The expected shape of the array.
        Raises:
        AssertionError: If the shape of the array does not match the expected shape.
        """
        assert arr.shape == expected_shape, f"Expected shape {expected_shape}, but got {arr.shape}"

    def predict(self, X):
        """
        Input:
        X test examples (n_examples x n_features)
        
        Ouput:
        Predicted class y_hat of each training example (n_examples)
        """
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        z1 = np.dot(X, self.W1.T) + self.b1 
        h1 = self.relu(z1)
        
        z2 = np.dot(h1, self.W2.T) + self.b2
        y_hat = self.softmax(z2)

        # Testing if the sum of each example is equal to one
        np.testing.assert_allclose(y_hat.sum(axis=1), np.ones(y_hat.shape[0]), rtol=1e-5)

        # Testing shape
        self.assert_shape(y_hat, (X.shape[0], self.n_classes))  # This should pass without raising an AssertionError

        return y_hat # n_examples x n_classes

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Dont forget to return the loss of the epoch.
        """

        y_hat = []
        for xi,yi in zip(X, y): # for each test object. There is probably a parallelized way of doing this
            h0 = xi
            
            z1 = self.W_1.T.dot(h0) + self.b_1 # I don't know if this works as intended
            h1 = np.maximum(0, z1)
            
            z2 = self.W_2.T.dot(h1) + self.b_2
            p = np.exp(z2) / sum(np.exp(z2))

            loss = -y.dot(np.log(p))

            
            grad_z2 = p - yi
            
            grad_W2 = grad_z2[:, None].dot(h1[:, None].T)
            grad_b2 = grad_z2
            
            # !! The equation for the hidden layer must change. Here I'm using as if the activation function was the tanh

            grad_h1 = self.W_2.T.dot(grad_z2)
            grad_z1 = grad_h1 * (1-h1**2)

            grad_W1 = grad_z1[:, None].dot(h0[:, None].T)
            grad_b1 = grad_z1

            self.W1 -= learning_rate*grad_W1
            self.b1 -= learning_rate*grad_b1
            self.W2 -= learning_rate*grad_W2
            self.b2 -= learning_rate*grad_b2
        
        return loss # Returns last loss

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

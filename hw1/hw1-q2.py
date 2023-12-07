#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
import time

import utils


# Q2.1
class LogisticRegression(nn.Module):

    def __init__(self, n_classes, n_features, **kwargs):
        """
        n_classes (int)
        n_features (int)
        """
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples
        """
        return self.linear(x)


# Q2.2
class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """
        n_classes (int)
        n_features (int)
        hidden_size (int)
        layers (int)
        activation_type (str)
        dropout (float): dropout probability
        """
        super().__init__()

        # Define the activation function
        if activation_type == 'relu':
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        # Create a list of modules
        modules = []
        in_features = n_features

        # Add hidden layers
        for _ in range(layers):
            modules.append(nn.Linear(in_features, hidden_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))
            in_features = hidden_size

        # Add output layer
        modules.append(nn.Linear(hidden_size, n_classes))

        # Combine all modules into a single sequential model
        self.model = nn.Sequential(*modules)

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples
        """
        # Forward pass through the sequential model
        return self.model(x)


def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()

    # Forward pass: Compute predicted y by passing X to the model
    y_hat = model(X)

    # Compute and print loss
    loss = criterion(y_hat, y)

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()

    return loss.item()


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


@torch.no_grad()
def evaluate(model, X, y, criterion):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    logits = model(X)
    loss = criterion(logits, y)
    loss = loss.item()
    y_hat = logits.argmax(dim=-1)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return loss, n_correct / n_possible


def plot(epochs, plottables, name='', ylim=None):
    """Plot the plottables over the epochs.
    
    Plottables is a dictionary mapping labels to lists of values.
    """
    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=1, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-hidden_size', type=int, default=100)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_oct_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))

    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]  # 10
    n_feats = dataset.X.shape[1]

    # initialize the model
    if opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = FeedforwardNetwork(
            n_classes,
            n_feats,
            opt.hidden_size,
            opt.layers,
            opt.activation,
            opt.dropout
        )

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.l2_decay)

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_losses = []
    valid_losses = []
    valid_accs = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        epoch_train_losses = []
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            epoch_train_losses.append(loss)

        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        print('Training loss: %.4f' % epoch_train_loss)
        print('Valid acc: %.4f' % val_acc)

        train_losses.append(epoch_train_loss)
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)

    _, test_acc = evaluate(model, test_X, test_y, criterion)
    print('Final Test acc: %.4f' % (test_acc))
    # plot
    if opt.model == "logistic_regression":
        config = (
            f"batch-{opt.batch_size}-lr-{opt.learning_rate}-epochs-{opt.epochs}-"
            f"l2-{opt.l2_decay}-opt-{opt.optimizer}"
        )
    else:
        config = (
            f"batch-{opt.batch_size}-lr-{opt.learning_rate}-epochs-{opt.epochs}-"
            f"hidden-{opt.hidden_size}-dropout-{opt.dropout}-l2-{opt.l2_decay}-"
            f"layers-{opt.layers}-act-{opt.activation}-opt-{opt.optimizer}"
        )

    losses = {
        "Train Loss": train_losses,
        "Valid Loss": valid_losses,
    }
    # Choose ylim based on model since logistic regression has higher loss
    if opt.model == "logistic_regression":
        ylim = (0., 1.6)
    elif opt.model == "mlp":
        ylim = (0., 1.2)
    else:
        raise ValueError(f"Unknown model {opt.model}")
    plot(epochs, losses, name=f'{opt.model}-training-loss-{config}', ylim=ylim)
    accuracy = { "Valid Accuracy": valid_accs }
    plot(epochs, accuracy, name=f'{opt.model}-validation-accuracy-{config}', ylim=(0., 1.))

    stop_time = time.time()
    print(f"------ Elapsed time: {stop_time - start_time} seconds ------")

if __name__ == '__main__':
    main()

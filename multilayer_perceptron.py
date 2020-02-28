# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    multilayer_perceptron.py                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mfiguera <mfiguera@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/14 12:30:53 by mfiguera          #+#    #+#              #
#    Updated: 2020/02/28 11:06:09 by mfiguera         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt

from model import Model

np.random.seed(42)

def scale(X):
    return X/X.max(axis=0)


def categorize(y, categories):
    for id_, name in enumerate(categories):
        y[y == name] = id_
    return y


def stratified_shuffle_split(full, val_size):
    ones = full[full[:, -1] == 1]
    zeros = full[full[:, -1] == 0]
    ratio = len(ones) / len(full)
    
    n_ones = int(val_size * ratio // 1)
    n_zeros = int(val_size * (1-ratio) // 1)
    while n_ones + n_zeros < val_size:
        n_zeros += 1
    
    train = np.concatenate((ones[:-n_ones], zeros[:-n_zeros]))
    np.random.shuffle(train)
    
    val = np.concatenate((ones[-n_ones:], zeros[-n_zeros:]))
    np.random.shuffle(val)
    
    return train, val


def plot_logs(train_log, val_log, cost_log):
    plt.subplot(211)
    plt.title("Accuracy")
    plt.ylabel("%")
    plt.xlabel("epoch")
    plt.plot(train_log, label='Train accuracy')
    plt.plot(val_log, label='Validation accuracy')
    plt.legend()
    plt.grid()
    
    plt.subplot(212)
    plt.title("Loss")
    plt.ylabel("cross-entropy")
    plt.xlabel("epoch")
    plt.plot(range(len(cost_log)), cost_log, label='Cost')
    plt.grid()
    plt.show()


def multilayer_perceptron(datafile, labels, val_split, savefile=None, logs=False, n_epochs=100, batch_size=1, dynamic_lr=True, learning_rate=0.01, visual=False):
    assert val_split > 0 and val_split < 1, "val_split need have a value between 0 and 1."
    
    try:
        data = pd.read_csv(datafile)
    except pd.errors.EmptyDataError:
        print ("Empty data file.")
        sys.exit(-1)
    except pd.errors.ParserError:
        print ("Error parsing file, needs to be a well formated csv.")
        sys.exit(-1)
    X = scale(data.to_numpy()[:,2:]).astype(float)
    y = categorize(data["diagnosis"].to_numpy().copy(), labels)
    full = np.concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
    train, val = stratified_shuffle_split(full, int(X.shape[0] * val_split))
    X_train, y_train = train[:, :-1].astype(float), train[:, -1:].reshape((len(train))).astype(float)
    X_val, y_val = val[:, :-1].astype(float), val[:, -1:].reshape((len(val))).astype(float)
    
    assert batch_size > 0 and n_epochs > 0, "batch_size and n_epochs need to be greater than 0."
    assert batch_size <= X_train.shape[0], f"batch_size ({batch_size}) needs to be smaller than number of training examples ({X_train.shape[0]})."

    classifier = Model((X.shape[1], 5, 5, 2))
    cost_log, train_log, val_log = classifier.train(X_train, y_train, X_val, y_val, n_epochs=n_epochs, batch_size=batch_size, dynamic_lr=dynamic_lr, lr=learning_rate, visual=visual)

    print(f"Final validation accuracy: [ {val_log[-1]} ]")

    if savefile:
        classifier.save_to_file(savefile)

    if logs:
        plot_logs(cost_log, train_log, val_log)



def parse_args():
    def positive_int(x):
        x = int(x)
        if x <= 0:
            raise argparse.ArgumentTypeError("Parameter needs to be greater than 0")
        return x

    def positive_float(x):
        x = float(x)
        if x <= 0:
            raise argparse.ArgumentTypeError("Parameter needs to be greater than 0")
        return x

    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="path to csv containg data to be trained on", type=str)
    parser.add_argument("--save", "-s", help="name of network save file", type=str, default=None, metavar="FILENAME")
    parser.add_argument("--val_split", "-vs", help="percentage of data dedicated to validaton",
        type=float, metavar="(1,99)", default=8, choices=range(1, 100))
    parser.add_argument("--plot", help="plot learning stats after training", action='store_true')
    parser.add_argument("--n_epochs", "-ne", help="number of epochs used in training", type=positive_int, default=100)
    parser.add_argument("--batch_size", "-bs", help="batch size used in training", type=positive_int, default=1)
    parser.add_argument("--dynamic_lr", "-dlr", help="toggle dynamic learning rate", action='store_true')
    parser.add_argument("--learning_rate", "-lr", help="starter learning rate", type=positive_float, default=0.01)
    parser.add_argument("--visualizer", "-v", help="toggle training visualizer", action='store_true')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    multilayer_perceptron(args.datafile, ['B', 'M'], args.val_split / 100, args.save, args.plot, args.n_epochs, args.batch_size, args.dynamic_lr, args.learning_rate, args.visualizer)
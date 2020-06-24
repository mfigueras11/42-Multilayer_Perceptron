# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    multilayer_perceptron.py                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mfiguera <mfiguera@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/14 12:30:53 by mfiguera          #+#    #+#              #
#    Updated: 2020/06/20 10:17:03 by mfiguera         ###   ########.fr        #
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



def plot_logs(train_log, val_log, cost_log, lr_log):
    plt.subplot(311)
    plt.title("Accuracy")
    plt.ylabel("%")
    plt.plot(train_log, label='Train')
    plt.plot(val_log, label='Validation')
    plt.legend()
    plt.grid()
    
    plt.subplot(312)
    plt.title("Loss")
    plt.ylabel("cross-entropy")
    plt.plot(range(len(cost_log)), cost_log, label='Cost')
    plt.grid()

    plt.subplot(313)
    plt.title("Learning rate")
    plt.plot(range(len(lr_log)), lr_log, label='lr')
    axes = plt.gca()
    axes.set_ylim([0 , lr_log[0] * 1.1])
    plt.xlabel("epoch")

    plt.grid()

    plt.show()



def one_hot(data, n):
    ret = np.zeros((len(data), n))
    for i, val in enumerate(data):
        ret[i, val] = 1
    return ret



def open_datafile(datafile):
    try:
        data = pd.read_csv(datafile)
    except pd.errors.EmptyDataError:
        print ("Empty data file.")
        sys.exit(-1)
    except pd.errors.ParserError:
        print ("Error parsing file, needs to be a well formated csv.")
        sys.exit(-1)
    return data


def multilayer_perceptron(args, labels=['B', 'M']):
    data = open_datafile(args.datafile)
    
    X = scale(data.to_numpy()[:,2:]).astype(float)
    y = one_hot(categorize(data["diagnosis"].to_numpy().copy(), labels), len(labels))
    full = np.concatenate((X, y.reshape(y.shape[0], len(labels))), axis=1)
    train, val = stratified_shuffle_split(full, int(X.shape[0] * args.val_split / 100))
    X_train, y_train = train[:, :-len(labels)].astype(float), train[:, -len(labels):].astype(float)
    X_val, y_val = val[:, :-len(labels)].astype(float), val[:, -len(labels):].astype(float)
    
    assert args.batch_size > 0 and args.n_epochs > 0, "batch_size and n_epochs need to be greater than 0."
    assert args.batch_size <= X_train.shape[0], f"batch_size ({args.batch_size}) needs to be smaller than number of training examples ({X_train.shape[0]})."

    classifier = Model((X.shape[1], 5, 5, 2))
    cost_log, train_log, val_log, lr_log = classifier.train(X_train, y_train, X_val, y_val, n_epochs=args.n_epochs, batch_size=args.batch_size, dynamic_lr=args.dynamic_lr
, lr=args.learning_rate, visual=args.visualizer)

    print(f"Final validation accuracy: [ {val_log[-1]} ]")
    print(f"Cross-Entropy at last step: [ {classifier.softmax_crossentropy_logits(classifier.forward(X_val)[-1], y_val)} ]")

    classifier.save_to_file(name=args.out)

    if args.plot:
        plot_logs(train_log, val_log, cost_log, lr_log)



def predict(args):
    model = Model(args.model)


def set_parser():
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
    subparsers = parser.add_subparsers()
    
    trainer = subparsers.add_parser("train")
    trainer.set_defaults(func=multilayer_perceptron)
    trainer.add_argument("datafile", help="path to csv containg data to be trained on", type=str)
    trainer.add_argument("--out", help="name of network save file", type=str, default="network.model", metavar="FILENAME")
    trainer.add_argument("--val_split", "-vs", help="percentage of data dedicated to validaton",
        type=float, metavar="(1,99)", default=20, choices=range(1, 100))
    trainer.add_argument("--plot", help="plot learning stats after training", action='store_true')
    trainer.add_argument("--n_epochs", "-ne", help="number of epochs used in training", type=positive_int, default=100)
    trainer.add_argument("--batch_size", "-bs", help="batch size used in training", type=positive_int, default=1)
    trainer.add_argument("--dynamic_lr", "-dlr", help="toggle dynamic learning rate", action='store_true')
    trainer.add_argument("--learning_rate", "-lr", help="starter learning rate", type=positive_float, default=0.01)
    trainer.add_argument("--visualizer", "-v", help="toggle training visualizer", action='store_true')
    
    predictor = subparsers.add_parser("predict")
    predictor.set_defaults(func=predict)
    predictor.add_argument("--data", help="path to csv containg the data we want to make predictions for. Needs to have the right format", type=str, required=True)
    predictor.add_argument("model", help="path to pretrained model pickle file", type=str)
    
    return parser


if __name__ == "__main__":
    parser = set_parser()
    args = parser.parse_args()
    try:
        func = args.func
    except:
        parser.error("too few arguments")
    func(args)
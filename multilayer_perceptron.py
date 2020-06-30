# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    multilayer_perceptron.py                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mfiguera <mfiguera@student.42.us.org>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/14 12:30:53 by mfiguera          #+#    #+#              #
#    Updated: 2020/06/30 14:43:27 by mfiguera         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
from os import path, mkdir
import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt

from model import Model
from config import Config as config

np.random.seed(42)

def set_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    
    trainer = subparsers.add_parser("train")
    trainer.set_defaults(func=multilayer_perceptron)
    trainer.add_argument("datafile", help="path to csv containg data to be trained on", type=str)
    trainer.add_argument("--out", help="name of network save file", type=str, default="network.model", metavar="FILENAME")
    val_group = trainer.add_mutually_exclusive_group(required=True)
    val_group.add_argument("--val_split", help="percentage of data dedicated to validaton", type=float, metavar="(1,99)", default=None, choices=range(1, 100))
    val_group.add_argument("--val_data", help="path to csv containing data to be used in validation", type=str)
    trainer.add_argument("--plot", help="plot learning stats after training", action='store_true')
    
    predictor = subparsers.add_parser("predict")
    predictor.set_defaults(func=predict)
    predictor.add_argument("model", help="path to pretrained model pickle file", type=str)
    predictor.add_argument("--data", help="path to csv containg the data we want to make predictions for. Needs to have the right format", type=str, required=True)
    predictor.add_argument("--validation", help="data is labeled and want to perform a validation exercice", action="store_true")
    predictor.add_argument("--save", help="outputs a .csv containing the data with the predictions aggregated", action="store_true")

    splitter = subparsers.add_parser("split")
    splitter.set_defaults(func=split)
    splitter.add_argument("datafile", help="path to csv containg data to be split", type=str)
    splitter.add_argument("--output_dir", help="path to csv containg data to be split", type=str, default="data")
    splitter.add_argument("--val_split", help="percentage of data dedicated to validaton", type=float, metavar="(1,99)", default=20, choices=range(1, 100))
    
    return parser



def multilayer_perceptron(args):
    data = open_datafile(args.datafile)
    
    raw_X = data.to_numpy()[:,2:].astype(float)
    classifier = Model((raw_X.shape[1], 5, 5, 2), config)
    y = data["diagnosis"].to_numpy().copy()
    full = np.concatenate((raw_X, y.reshape(y.shape[0], 1)), axis=1)

    train, val = None, None

    if args.val_split:
        train, val = stratified_shuffle_split(full, args.val_split)
    else:
        val_data = open_datafile(args.val_data)
        val = val_data.to_numpy()[:, 2:].astype(float)
        val = np.concatenate((val, val_data["diagnosis"].to_numpy().copy().reshape(len(val), 1)), axis=1)
        train = full

    X_train, y_train = train[:, :-1].astype(float), one_hot(categorize(train[:, -1], config.labels), len(config.labels)).astype(float)
    X_val, y_val = val[:, :-1].astype(float), one_hot(categorize(val[:, -1], config.labels), len(config.labels)).astype(float)

    _ = classifier.scale_data(np.concatenate((X_train, X_val), axis=0))
    X_train = classifier.scale_data(X_train)
    X_val = classifier.scale_data(X_val)

    assert config.batch_size > 0 and config.epoch_number > 0, "batch_size and n_epochs need to be greater than 0."
    assert config.batch_size <= X_train.shape[0], f"batch_size ({config.batch_size}) needs to be smaller than number of training examples ({X_train.shape[0]})."

    cost_log, train_log, val_log, lr_log = classifier.train(X_train, y_train, X_val, y_val)

    run_validation(classifier.predict(X_val), y_val)
    
    classifier.save_to_file(name=args.out)

    if args.plot:
        plot_logs(train_log, val_log, cost_log, lr_log)



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



def predict(args):

    model = Model(args.model, NotImplemented)
    data = open_datafile(args.data)
    
    X = model.scale_data(data.to_numpy()[:,2:]).astype(float)
    preds = model.predict(X)
    data["predictions"] = [config.labels[p] for p in preds]
    if args.validation:
        y = one_hot(categorize(data["diagnosis"].to_numpy().copy(), config.labels), len(config.labels))
        logits = model.forward(X)[-1]
        print(f"Crossentropy at last step: {model.softmax_crossentropy_logits(logits[:,-1], y[:, -1])}")
        run_validation(preds, y)
    
    if args.save:
        save_to_file(data)



def split(args):
    data = open_datafile(args.datafile)
    train, val = stratified_shuffle_split(data.to_numpy(), args.val_split, label_index=1)
    save_to_file(pd.DataFrame(train, columns=data.columns), directory="data", name="training.csv")
    save_to_file(pd.DataFrame(val, columns=data.columns), directory="data", name="validation.csv")



def stratified_shuffle_split(data, val_split, label_index=-1):
    np.random.shuffle(data)
    categories = [data[data[:, label_index] == l] for l in config.labels]

    ratios = [int(len(cat) * val_split // 100) for cat in categories]

    train = np.concatenate([cat[:-ratios[i]] for i, cat in enumerate(categories)])
    np.random.shuffle(train)

    val = np.concatenate([cat[-ratios[i]:] for i, cat in enumerate(categories)])
    np.random.shuffle(val)

    return train, val



def run_validation(predictions, val_data):
    predicted = one_hot(predictions, val_data.shape[1])
    for i, label in enumerate(config.labels):
        preds = predicted[:, i]
        y = val_data[:, i]
        
        true_negative = np.sum((preds == 0) * (y == 0))
        true_positives = np.sum((preds == 1) * (y == 1))
        false_negative = np.sum(preds < y)
        false_positive = np.sum(preds > y)
        n_examples = len(preds)

        if n_examples == 0:
            accuracy = "ERROR"
        else:
            accuracy = (true_negative + true_positives) / n_examples

        if true_positives + false_positive == 0:
            precision = 0
        else:
            precision = true_positives / (true_positives + false_positive)

        if true_positives + false_negative == 0:
            recall = 0
        else:
            recall = true_positives / (true_positives + false_negative)

        if precision + recall == 0:
            F1 = 0
        else:
            F1 = 2 * (precision * recall) / (precision + recall)

        print(f"Label {label}: Accuracy={accuracy:.3f} Precision={precision:.3f} Recall={recall:.3f} F1={F1:.3f}")





def save_to_file(dataframe, directory="predictions", name="prediction.csv", n=0):
    def get_file_name(name, n):
        if n:
            extension = name.split('.')[-1]
            name = ".".join(name.split('.')[:-1])
            name = name + " (" + str(n) +")."+extension
        return name

    if not name.endswith('.csv'):
        name += ".csv"
    filename = get_file_name(directory + "/" + name, n)
    if not path.exists(directory) or not path.isdir(directory):
        try:
            mkdir(directory)
        except:
            print("Error creating subdirectory. File could not be saved.")
            return
    if path.exists(filename):
        return save_to_file(dataframe, directory, name, n+1)
    with open(filename, "w+") as file:
        dataframe.to_csv(file, index=False)
        print(f"{path.splitext(name)[0].capitalize()} was saved in file: {filename}")
    return filename



def categorize(y, categories):
    for id_, name in enumerate(categories):
        y[y == name] = id_
    return y



def one_hot(data, n):
    ret = np.zeros((len(data), n))
    for i, val in enumerate(data.flat):
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



if __name__ == "__main__":
    parser = set_parser()
    args = parser.parse_args()
    try:
        func = args.func
    except:
        parser.error("too few arguments")
    func(args)

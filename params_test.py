# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    multilayer_perceptron.py                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mfiguera <mfiguera@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/14 12:30:53 by mfiguera          #+#    #+#              #
#    Updated: 2020/02/21 08:39:57 by mfiguera         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import datetime
import numpy as np
import pandas as pd

from model import Model

np.random.seed(42)

def scale(X, xmax=None):
    if not xmax:
        xmax = X.max(axis=0)
    return X/xmax

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


data = pd.read_csv("resources/data.csv")
X = scale(data.to_numpy()[:,2:]).astype(float)
assert "diagnosis" in data
y = categorize(data["diagnosis"].to_numpy().copy(), ['B', 'M'])
full = np.concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
train, val = stratified_shuffle_split(full, 100)
X_train, y_train = train[:, :-1].astype(float), train[:, -1:].reshape((len(train))).astype(float)
X_val, y_val = val[:, :-1].astype(float), val[:, -1:].reshape((len(val))).astype(float)


df = pd.DataFrame(columns=["acc", "n_epoch", "shuffle", "batch_size", "dynamic_lr", "cost_log", "train_log", "val_log"])
for n_epoch in [10, 100, 250]:
    for batch_size in [1, 2, 8, 16, 32]:
        for shuffle in [True, False]:
            for dynamic_lr in [True, False]:
                for hidden_s in [4, 5, 8]:
                    classifier = Model((30,hidden_s,hidden_s,2))
                    t0 = datetime.datetime.now()
                    cost_log, train_log, val_log = classifier.train(X_train, y_train, X_val, y_val, n_epoch=n_epoch, batch_size=batch_size, dynamic_lr=dynamic_lr, shuffle=shuffle, quiet=True)
                    t1 = datetime.datetime.now()
                    print(f"e: {n_epoch} bz: {batch_size} s: {str(shuffle)[0]} d: {str(dynamic_lr)[0]} acc: {classifier.score(classifier.predict(X_val), y_val)}")
                    filename = classifier.save_to_file(f"test2/e{n_epoch}_bz{batch_size}_s{str(shuffle)[0]}_d{str(dynamic_lr)[0]}_hs{hidden_s}.pickle")
                    d = {
                        "acc": classifier.score(classifier.predict(X_val), y_val),
                        "n_epoch": n_epoch,
                        "batch_size": batch_size,
                        "dynamic_lr": dynamic_lr,
                        "shuffle": shuffle,
                        "cost_log": cost_log,
                        "train_log": train_log,
                        "val_log": val_log,
                        "hidden_s": hidden_s,
                        "time": (t1-t0).total_seconds()
                    }
                    df = df.append(d, ignore_index=True)

df.to_csv("experiment.csv")
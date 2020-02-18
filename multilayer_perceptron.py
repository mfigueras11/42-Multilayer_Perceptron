# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    multilayer_perceptron.py                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mfiguera <mfiguera@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/14 12:30:53 by mfiguera          #+#    #+#              #
#    Updated: 2020/02/18 19:37:25 by mfiguera         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd

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


data = pd.read_csv("resources/data.csv")
X = scale(data.to_numpy()[:,2:]).astype(float)
y = categorize(data["diagnosis"].to_numpy().copy(), ['B', 'M'])
full = np.concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
train, val = stratified_shuffle_split(full, 75)
X_train, y_train = train[:, :-1].astype(float), train[:, -1:].reshape((len(train))).astype(float)
X_val, y_val = val[:, :-1].astype(float), val[:, -1:].reshape((len(val))).astype(float)

classifier = Model((X.shape[1], 4, 4, 2))
classifier.train(X_train, y_train, X_val, y_val, n_epoch=150, batch_size=1)
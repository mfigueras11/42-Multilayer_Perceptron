# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    multilayer_perceptron.py                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mfiguera <mfiguera@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/14 12:30:53 by mfiguera          #+#    #+#              #
#    Updated: 2020/02/17 12:29:12 by mfiguera         ###   ########.fr        #
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


data = pd.read_csv("resources/data.csv")
X = scale(data.to_numpy()[:,2:]).astype(float)
y = categorize(data["diagnosis"].to_numpy().copy(), ['B', 'M'])
X_train, y_train = X[:-50], y[:-50]
X_val, y_val = X[-50:], y[-50:]

classifier = Model((X.shape[1], 4, 4, 2))
classifier.train(X_train, y_train, X_val, y_val, n_epoch=200, batch_size=32)
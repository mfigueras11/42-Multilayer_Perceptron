# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    model.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mfiguera <mfiguera@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/14 09:36:15 by mfiguera          #+#    #+#              #
#    Updated: 2020/02/28 10:00:22 by mfiguera         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
from os import path
import numpy as np
import pickle as pk
from tqdm import trange

from layers import Dense, ReLU, Softmax, Sigmoid
import visualizer


class Model:
    def __init__(self, network, activation='relu', output='softmax'):
        self.network = self.__get_network(network, activation.lower(), output.lower())


    @staticmethod
    def __get_network(input_, activation, output):
        if type(input_) == tuple or type(input_) == list:
            layers = {'relu': ReLU, 'dense':Dense, 'softmax': Softmax, 'sigmoid': Sigmoid}
            network=[]
            n_units = input_
            n_layers = len(n_units) - 1
            for i in range(n_layers):
                network.append(Dense(*n_units[i:i+2]))
                if i + 1 < n_layers:
                    network.append(layers.get(activation, ReLU)())
                else:
                    network.append(layers.get(output, Softmax)())
        elif type(input_) == str:
            try:
                with open(input_, 'rb') as file:
                    network = pk.load(file)
            except Exception:
                print(f"File {input_} not found or corrupt.")
                sys.exit()
        return network
            

    def forward(self, X):
        activations = []

        input_ = X
        for l in self.network:
            activations.append(l.forward(input_))
            input_ = activations[-1]
        
        assert len(activations) == len(self.network), "Error here"
        
        return activations


    def predict(self, X):
        logits = self.forward(X)[-1]
        return logits.argmax(axis=-1)


    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        batch_size = kwargs.get("batch_size", 32)
        shuffle = kwargs.get("shuffle", True)
        n_epoch = kwargs.get("n_epochs", 25)
        
        lr = kwargs.get("lr", 0.01)
        dynamic_lr = kwargs.get("dynamic_lr", True)
        lr_multiplier = kwargs.get("lr_multiplier", 0.66)
        lr_epoch_change = kwargs.get("lr_epoch_change", 25)

        visual = kwargs.get("visual", False)

        train_log = []
        val_log = []
        cost_log = []

        if visual:
            visualizer.setup()

        for ep in trange(n_epoch):
            if dynamic_lr and ep and ep % lr_epoch_change == 0:
                lr *= lr_multiplier
            cost = 0
            act_visual = visual
            for X_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size, shuffle):
                cost+=self.train_step(X_batch, y_batch, lr, act_visual)
                act_visual = False

            train_log.append(self.score(self.predict(X_train), y_train))
            val_log.append(self.score(self.predict(X_val), y_val))
            cost_log.append(cost)

        return cost_log, train_log, val_log


    @staticmethod
    def iterate_minibatches(X, y, batch_size, shuffle):
        assert len(X) == len(y), "X and Y have different sizes"
        if shuffle:
            indices = np.random.permutation(len(y))
        for i in range(0, len(y) - batch_size + 1, batch_size):
            if shuffle:
                selection = indices[i:i+batch_size]
            else:
                selection = slice(i, i+batch_size)
            yield X[selection], y[selection]


    def train_step(self, X, y, lr, visual=False):
        activations = self.forward(X)
        inputs = [X] + activations
        logits = activations[-1]

        loss = self.softmax_crossentropy_logits(logits, y)

        loss_grad = self.network[-1].grad(logits, y)
        
        if visual:
            visualizer.draw_network(activations[1::2], 30)

        for i in range(len(self.network[:-1]))[::-1]:
            layer = self.network[i]
            loss_grad = layer.backward(inputs[i], loss_grad, lr)

        return np.mean(loss)


    @staticmethod
    def score(y_pred, y_true):
        return np.mean(y_pred==y_true)


    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)


    @staticmethod
    def softmax_crossentropy_logits(pred_logits, y):
        logits_for_answers = pred_logits[np.arange(len(pred_logits)), y.flatten().astype(int)]
        return np.log(np.sum(np.exp(pred_logits), axis=-1)) - logits_for_answers


    @staticmethod
    def gradient_softmax_crossentropy_logits(pred_logits, y):
        ones_for_answers = np.zeros_like(pred_logits)
        ones_for_answers[np.arange(len(pred_logits)), y.flatten().astype(int)] = 1
        return (pred_logits - ones_for_answers) / pred_logits.shape[0]


    @staticmethod
    def __get_file_name(name, n):
        if n:
            extension = name.split('.')[-1]
            name = ".".join(name.split('.')[:-1])
            name = name + "(" + str(n) +")."+extension
        return name


    def save_to_file(self, name="network.pickle", n=0):
        if not name.endswith('.pickle'):
            name += ".pickle"
        filename = self.__get_file_name(name, n)
        if path.exists(filename):
            return self.save_to_file(name, n+1)
        with open(filename, "wb+") as file:
            pk.dump(self.network, file)
            print(f"Network was saved in file: {filename:.2f}")
        return filename

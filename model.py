# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    model.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mfiguera <mfiguera@student.42.us.org>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/14 09:36:15 by mfiguera          #+#    #+#              #
#    Updated: 2020/06/30 12:20:25 by mfiguera         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
from os import path, mkdir
import numpy as np
import pickle as pk
from tqdm import trange

from layers import Dense
from activations import ReLU, Softmax, Sigmoid


class Model:
    def __init__(self, network, config):
        self.network = self.__get_network(network)
        self.test = True
        self.xmax = None
        self.xmin = None
        self.config = config


    def scale_data(self, X):
        if type(self.xmax):
            self.xmax = X.max(axis=0)
            self.xmin = X.min(axis=0)
        if len(self.xmax) != X.shape[1]:
            print("Data shape does not have the shape pretrained for this model.")
            sys.exit()
        return X/self.xmax


    def __get_network(self, input_):
        if type(input_) == tuple or type(input_) == list:
            activations = {'relu': ReLU, 'softmax': Softmax, 'sigmoid': Sigmoid}
            network=[]
            n_units = input_
            n_layers = len(n_units) - 1
            for i in range(n_layers):
                network.append(Dense(*n_units[i:i+2]))
                if i + 1 < n_layers:
                    network.append(activations.get(self.config.activation, ReLU)())
                else:
                    network.append(activations.get(self.config.output_activation, Softmax)())
        elif type(input_) == str:
            filename = input_
            if filename.endswith(".model"):
                try:
                    with open(filename, 'rb') as file:
                        network = pk.load(file)
                except Exception:
                    print(f"File {filename} not found or corrupt.")
                    sys.exit()
            else:
                print("Input file needs to be of a *.model type")
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


    def train(self, X_train, y_train, X_val, y_val):
        batch_size = self.config.batch_size
        shuffle = self.config.shuffle
        n_epoch = self.config.epoch_number
        
        lr = self.config.learning_rate
        dyn_lr = self.config.dynamic_learning_rate
        lr_multiplier = self.config.learning_rate_multiplier
        lr_n_changes = self.config.learning_rate_change_number
        lr_epoch_change = n_epoch // lr_n_changes

        train_log = []
        val_log = []
        cost_log = []
        lr_log = []
        
        t = trange(n_epoch)
        for ep in t:
            if dyn_lr and ep and ep % lr_epoch_change == 0:
                lr *= lr_multiplier
            cost = []
            for X_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size, shuffle):
                cost.append(self.train_step(X_batch, y_batch, lr))

            train_log.append(self.score(self.predict(X_train), y_train))
            val_log.append(self.score(self.predict(X_val), y_val))
            cost_log.append(np.mean(cost))
            lr_log.append(lr)
            t.set_description(f"Cost: {cost_log[-1]:10.10}")

        return cost_log, train_log, val_log, lr_log


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


    def train_step(self, X, y, lr):
        activations = self.forward(X)
        inputs = [X] + activations
        logits = activations[-1]

        loss = self.softmax_crossentropy_logits(logits[:,1], y[:,1])

        loss_grad = self.network[-1].grad(logits, y)
        
        for i in range(len(self.network) - 1)[::-1]:
            layer = self.network[i]
            loss_grad = layer.backward(inputs[i], loss_grad, lr)

        return loss


    def score(self, y_pred, y_true):
        return np.mean(y_pred == y_true[:, 1])


    @staticmethod
    def softmax_crossentropy_logits(pred_logits, y):
        if len(pred_logits.shape) == 1:
            pred_logits = pred_logits.reshape((pred_logits.shape[0], 1))
            y = y.reshape((y.shape[0], 1))
        a = y * np.log(pred_logits)
        b = (1 - y) * np.log(1 - pred_logits)
        c = (a + b).sum(axis=1)
        if -np.sum(c) / len(y) == np.NaN:
            sys.exit()

        return -np.sum(c) / len(y)



    @staticmethod
    def __get_file_name(name, n):
        if n:
            extension = name.split('.')[-1]
            name = ".".join(name.split('.')[:-1])
            name = name + " (" + str(n) +")."+extension
        return name


    def save_to_file(self, directory="networks", name="network.model", n=0):
        if not name.endswith('.model'):
            name += ".model"
        filename = self.__get_file_name(directory + "/" + name, n)
        if not path.exists(directory) or not path.isdir(directory):
            try:
                mkdir(directory)
            except:
                print("Error creating subdirectory. File could not be saved.")
                return
        if path.exists(filename):
            return self.save_to_file(directory, name, n+1)
        with open(filename, "wb+") as file:
            pk.dump(self.network, file)
            print(f"Network was saved in file: {filename}")
        return filename

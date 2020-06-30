# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    activations.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mfiguera <mfiguera@student.42.us.org>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/06/30 14:54:16 by mfiguera          #+#    #+#              #
#    Updated: 2020/06/30 15:01:56 by mfiguera         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from layers import Layer
import numpy as np

class Activation(Layer):
    pass



class ReLU(Activation):
    def __init__(self):
        pass


    def __str__(self):
        return "ReLU"


    def forward(self, input_):
        return np.maximum(1e-3 * input_, input_)


    def backward(self, input_, grad_output=None, lr=None):
        relu_grad = input_ > 0
        relu_grad = np.array([[1 if l else 1e-3 for l in i] for i in relu_grad])
        return grad_output * relu_grad



class Softmax(Activation):
    def __init__(self):
        pass


    def __str__(self):
        return "Softmax"


    def forward(self, input_):
        return self.softmax(input_)

    @staticmethod
    def grad(pred_logits, y):
        return (pred_logits - y) / pred_logits.shape[0]

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x))
        # print(exps)
        return exps / np.sum(exps, axis=0)



class Sigmoid(Activation):
    def __init__(self):
        pass


    def __str__(self):
        return "Sigmoid"


    def forward(self, input_):
        return 1 / (1 + np.exp(-input_))


    def backward(self, input_, grad_output=None, lr=None):
        return input_ * (1 - input_)

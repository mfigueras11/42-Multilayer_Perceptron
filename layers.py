# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    layers.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mfiguera <mfiguera@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/12 09:39:56 by mfiguera          #+#    #+#              #
#    Updated: 2020/02/25 11:37:31 by mfiguera         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np


class Layer:
    def __init__(self):
        pass


    def __str__(self):
        return "Layer"


    def forward(self, input_):
        return input_


    def backward(self, input_, grad_output):
        return grad_output



class ReLU(Layer):
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



class Softmax(Layer):
    def __init__(self):
        pass


    def __str__(self):
        return "Softmax"


    def forward(self, input_):
        return self.softmax(input_)

    @staticmethod
    def grad(pred_logits, y):
        reference = np.zeros_like(pred_logits)
        reference[np.arange(len(y)), y.flatten().astype(int)] = 1
        return (pred_logits - reference) / pred_logits.shape[0]

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)



class Sigmoid(Layer):
    def __init__(self):
        pass


    def __str__(self):
        return "Sigmoid"


    def forward(self, input_):
        return 1 / (1 + np.exp(-input_))


    def backward(self, input_, grad_output=None, lr=None):
        return input_ * (1 - input_)



class Dense(Layer):
    def __init__(self, in_units, out_units):
        self.weights = np.random.normal(size=(in_units, out_units), loc=0.0, scale=np.sqrt(2/(in_units + out_units))).astype(float)
        self.biases = np.zeros(out_units)


    def __str__(self):
        return f"Dense - {self.weights.shape}"


    def forward(self, input_):
        assert input_.shape[1] == self.weights.shape[0], f"Input {input_.shape} does not match Weights {self.weights.shape} shape."
        return input_ @ self.weights + self.biases


    def backward(self, input_, grad_output, lr=0.01):
        grad_input = grad_output @ self.weights.T

        grad_weights = input_.T @ grad_output
        grad_biases = grad_output.mean(axis=0)
        
        assert grad_weights.shape == self.weights.shape, f"Grad shape does not match weights shape ({grad_weights.shape} and {self.weights.shape})."
        assert grad_biases.shape == self.biases.shape, f"Grad shape does not match biasess shape ({grad_biases.shape} and {self.biases.shape})."

        self.weights -= lr * grad_weights
        self.biases -= lr * grad_biases

        return grad_input

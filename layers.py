# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    layers.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mfiguera <mfiguera@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/12 09:39:56 by mfiguera          #+#    #+#              #
#    Updated: 2020/02/14 09:47:44 by mfiguera         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np


class Layer:
    def __init__(self):
        pass
    

    def forward(self, input_):
        return input_


    def backward(self_, input_, grad_output):
        return grad_output



class ReLU(Layer):
    def __init__(self):
        pass


    def forward(self, input_):
        return np.maximum(0, input_)


    def backward(self, input_, grad_output):
        relu_grad = input_ > 0
        return grad_output * relu_grad



class Dense(Layer):
    def __init__(self, in_units, out_units, lr=0.1):
        self.lr = lr
        self.weigths = np.random.normal(size=(in_units, out_units), loc=0.0, scale=np.sqrt(2/(in_units + out_units)))
        self.biases = np.zeros(out_units)

    
    def forward(self, input_):
        assert input_.shape[1] != self.weigths.shape[0], "Input doesnt match Weights shape."
        return input_ @ self.weigths + self.biases

    
    def backward(self, input_, grad_output):
        grad_input = grad_output @ self.weigths.T

        grad_weights = input_.T @ grad_output
        grad_biases = grad_output.mean(axis=0)*input_.shape[0]
        
        assert grad_weights.shape == self.weigths.shape, f"Grad shape does not match weights shape ({grad_weights.shape} and {self.weigths.shape})."
        assert grad_biases.shape == self.biases.shape, f"Grad shape does not match biasess shape ({grad_biases.shape} and {self.biases.shape})."

        self.weigths -= self.lr * grad_weights
        self.biases -= self.lr * grad_biases

        return grad_input

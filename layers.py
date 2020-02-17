# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    layers.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mfiguera <mfiguera@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/12 09:39:56 by mfiguera          #+#    #+#              #
#    Updated: 2020/02/17 18:18:14 by mfiguera         ###   ########.fr        #
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
        # print(f"F {self.__class__.__name__}: {input_}")
        return np.maximum(1e-9, input_)


    def backward(self, input_, grad_output, lr=None):
        # print(f"B {self.__class__.__name__} grad: {grad_output} in: {input_}")
        relu_grad = input_ > 0
        return grad_output * relu_grad + 1e-9



class Dense(Layer):
    def __init__(self, in_units, out_units):
        self.weights = np.random.normal(size=(in_units, out_units), loc=0.0, scale=np.sqrt(2/(in_units + out_units))).astype(float)
        self.biases = np.zeros(out_units)

    
    def forward(self, input_):
        # print(f"F {self.__class__.__name__}: {input_}")
        assert input_.shape[1] == self.weights.shape[0], f"Input {input_.shape} does not match Weights {self.weights.shape} shape."
        return input_ @ self.weights + self.biases

    
    def backward(self, input_, grad_output, lr=0.01):
        grad_input = grad_output @ self.weights.T

        # print(f"B {self.__class__.__name__} grad: {grad_output} in: {input_}")

        grad_weights = input_.T @ grad_output
        grad_biases = grad_output.mean(axis=0)*input_.shape[0]
        
        assert grad_weights.shape == self.weights.shape, f"Grad shape does not match weights shape ({grad_weights.shape} and {self.weights.shape})."
        assert grad_biases.shape == self.biases.shape, f"Grad shape does not match biasess shape ({grad_biases.shape} and {self.biases.shape})."

        self.weights -= lr * grad_weights
        self.biases -= lr * grad_biases

        return grad_input

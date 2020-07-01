# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    config.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mfiguera <mfiguera@student.42.us.org>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/06/29 12:43:59 by marti             #+#    #+#              #
#    Updated: 2020/07/01 19:29:18 by mfiguera         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from activations import ReLU, Softmax, Sigmoid

class Config:
    labels = ['B', 'M']                     ### List containing values of categories.
    epoch_number = 100                      ### Number of epochs expected in training.
    
    batch_size = 1                          ### DEPRECATED - Batch size != 1 should not be used.
    shuffle = True                          ### Shuffle dataset during training. In general it is better to have
                                            #   shuffled data for the perfomance of a model.

    activation = ReLU                       ### Activation layer for neural net.
    output_activation = Softmax             ### Last layer activation.
    
    learning_rate = 0.01                    ### Learning rate at beggingin of training.
    dynamic_learning_rate = True            ### Learning rate will decrease with time to allow for more
                                            #   change at the beggining and more precision at the end of training.
    learning_rate_multiplier = 0.666        ### Number that will multiply learning rate at each change.
    learning_rate_change_number = 6         ### How many changes of LR will there be during training.


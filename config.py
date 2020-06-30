# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    config.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mfiguera <mfiguera@student.42.us.org>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/06/29 12:43:59 by marti             #+#    #+#              #
#    Updated: 2020/06/30 12:14:14 by mfiguera         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

class Config:
    labels = ['B', 'M']
    epoch_number = 100
    batch_size = 1
    learning_rate = 0.01
    shuffle = True
    dynamic_learning_rate = True
    learning_rate_multiplier = 0.666
    learning_rate_change_number = 6


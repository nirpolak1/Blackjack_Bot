# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 19:37:42 2020

@author: user
"""

import numpy as np


def ann(parameters, player_cards, dealer_card):
    input_layer = np.append(player_cards, dealer_card)
    first_hidden_size = 10
    second_hidden_size = 10
    
    weights = parameters[:(input_layer.shape[0]*first_hidden_size+first_hidden_size*second_hidden_size+second_hidden_size*2)]
    bias = parameters[(input_layer.shape[0]*first_hidden_size+first_hidden_size*second_hidden_size+second_hidden_size*2):]
    
    first_hidden_weights = weights[0:(first_hidden_size*input_layer.shape[0])]
    first_hidden_weights = np.reshape(first_hidden_weights, (input_layer.shape[0],first_hidden_size))
    
    first_hidden_layer = input_layer.dot(first_hidden_weights) + bias[:first_hidden_size]
    first_hidden_layer = swish(first_hidden_layer)
    

    second_hidden_weights = weights[(first_hidden_size*input_layer.shape[0]):((first_hidden_size*input_layer.shape[0])+(second_hidden_size*first_hidden_size))]
    second_hidden_weights = np.reshape(second_hidden_weights, (first_hidden_size,second_hidden_size))
    
    second_hidden_layer = first_hidden_layer.dot(second_hidden_weights) + bias[first_hidden_size:first_hidden_size+second_hidden_size]
    second_hidden_layer = swish(second_hidden_layer)
    
    output_weights = weights[((first_hidden_size*input_layer.shape[0])+(second_hidden_size*first_hidden_size)):]
    output_weights = np.reshape(output_weights, (second_hidden_size,2))
    output_value = second_hidden_layer.dot(output_weights) + bias[first_hidden_size+second_hidden_size:]
    return output_value

def swish(x):
    return (x/(1+np.exp(-x)))

def leakyReLU(x):
    return np.where(x>0,x,0.01*x)

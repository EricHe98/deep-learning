from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools as it
import pandas as pd

def relu(x):
    return x.clamp(min=0)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def identity(x):
    return x

def d_identity(x):
    return torch.ones_like(x)

def d_sigmoid(x):
    return x * (1 - x)

def d_relu(x):
    return torch.gt(x, 0).int()

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()
        
        self.function_map = {'relu':relu,
                             'sigmoid':sigmoid,
                             'identity':identity}
        
        self.gradient_map = {'relu':d_relu,
                             'sigmoid':d_sigmoid,
                             'identity':d_identity}
        
        self.f = self.function_map[self.f_function]
        self.g = self.function_map[self.g_function]
        
        self.d_f = self.gradient_map[self.f_function]
        self.d_g = self.gradient_map[self.g_function]

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        self.cache['batch_size'] = x.shape[0]
        self.cache['x'] = x
        self.cache['s_1'] = (self.parameters['W1'] @ x.T).T + self.parameters['b1']
        self.cache['z_1'] = self.f(self.cache['s_1'])
        self.cache['s_2'] = (self.parameters['W2'] @ self.cache['z_1'].T).T + self.parameters['b2']
        self.cache['z_2'] = self.g(self.cache['s_2'])
        return self.cache['z_2']
        
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        self.cache['dz2ds2'] = self.d_g(self.cache['z_2']) # (batch_size, linear_2_out_features)
        self.cache['ds2dW2'] = self.cache['z_1'] # (batch_size, linear_2_out_features, )
        self.cache['ds2db2'] = torch.tensor([1.])
        self.cache['ds2dz1'] = self.parameters['W2']
        self.cache['dz1ds1'] = self.d_f(self.cache['z_1'])
        self.cache['ds1db1'] = torch.tensor([1.])
        self.cache['ds1dw1'] = self.cache['x']
        
        self.grads['dJdW2'] = (dJdy_hat * self.cache['dz2ds2']).T @ self.cache['ds2dW2'] \
            / self.cache['batch_size']
        self.grads['dJdb2'] = ((dJdy_hat * self.cache['dz2ds2']).T * self.cache['ds2db2']).sum(axis=1) \
            / self.cache['batch_size']
        self.grads['dJdW1'] = (dJdy_hat \
            * self.cache['dz2ds2'] \
            @ self.cache['ds2dz1'] \
            * self.cache['dz1ds1']).T \
            @ self.cache['ds1dw1'] \
            / self.cache['batch_size']
        self.grads['dJdb1'] = ((dJdy_hat \
            * self.cache['dz2ds2'] \
            @ self.cache['ds2dz1'] \
            * self.cache['dz1ds1']).T \
            * self.cache['ds1db1']) \
            .sum(axis=1) \
            / self.cache['batch_size']
    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    loss = (0.5 * (y - y_hat) ** 2).mean()
    dJdy_hat = y_hat - y
    return loss, dJdy_hat


    # return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    loss = -(y * torch.log(y_hat) + (1. - y) * torch.log(1. - y_hat)).mean()
    dJdy_hat = (y_hat - y) / (y_hat * (1 - y_hat))
    return loss, dJdy_hat








from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP, mse_loss, bce_loss

import itertools as it
import pandas as pd

def test(f,g,loss):
    net = MLP(
        linear_1_in_features=2,
        linear_1_out_features=20,
        f_function=f,
        linear_2_in_features=20,
        linear_2_out_features=1,
        g_function=g
    )
    x = torch.randn(10, 2)
    y = ((torch.randn(10) > 0.5) * 1.0).unsqueeze(-1)
    
    net.clear_grad_and_cache()
    y_hat = net.forward(x)
    if loss == 'mse':
        J, dJdy_hat = mse_loss(y, y_hat)
        net.backward(dJdy_hat)
    elif loss == 'bce' and g == 'sigmoid':
        J, dJdy_hat = bce_loss(y, y_hat)
        net.backward(dJdy_hat)
    else:
        return ([('remove inf/nan bce'),0.0,0.0,0.0,0.0,0.0])
    
    #------------------------------------------------
    # check the result with autograd
    set_function = dict(
            relu     = nn.ReLU(),
            sigmoid  = nn.Sigmoid(),
            identity = nn.Identity()
    )
    
    net_autograd = nn.Sequential(
        OrderedDict([
            ('linear1', nn.Linear(2, 20)),
            (f, set_function[f]),
            ('linear2', nn.Linear(20, 1)),
            (g+'2', set_function[g])
        ])
    )
    net_autograd.linear1.weight.data = net.parameters['W1']
    net_autograd.linear1.bias.data = net.parameters['b1']
    net_autograd.linear2.weight.data = net.parameters['W2']
    net_autograd.linear2.bias.data = net.parameters['b2']
    
    y_hat_autograd = net_autograd(x)
    if loss == 'mse':
        J_autograd = 0.5 * F.mse_loss(y_hat_autograd, y)
    elif loss == 'bce' and g == 'sigmoid':
        J_autograd = F.binary_cross_entropy(y_hat_autograd, y)
    else:
        raise("shouldn't be here")
    
    net_autograd.zero_grad()
    J_autograd.backward()
    
    # print('dJdW1', net.grads['dJdW1'])
    # print(net_autograd.linear1.weight.grad.data)
    # print('dJdb1', net.grads['dJdb1'])
    # print(net_autograd.linear1.bias.grad.data)
    # print('dJdW2', net.grads['dJdW2'])
    # print(net_autograd.linear2.weight.grad.data)
    # print('dJdb2', net.grads['dJdb2'])
    # print(net_autograd.linear2.bias.grad.data)
    
    a = (net_autograd.linear1.weight.grad.data - net.grads['dJdW1']).norm().float().item()
    b = (net_autograd.linear1.bias.grad.data - net.grads['dJdb1']).norm().float().item()
    c = (net_autograd.linear2.weight.grad.data - net.grads['dJdW2']).norm().float().item()
    d = (net_autograd.linear2.bias.grad.data - net.grads['dJdb2']).norm().float().item()
    if loss == 'mse':
        e = (J_autograd - J).norm().float().item()
    else:
        if (torch.isinf(J_autograd).any() or torch.isnan(J_autograd).any() or
            torch.isinf(J).any() or torch.isnan(J).any()):
            return ([('remove inf/nan bce'),0.0,0.0,0.0,0.0,0.0])
        e = (J_autograd - J).norm().float().item()
    return([(f,g,loss),a,b,c,d,e])

    #------------------------------------------------

NUM_TESTS = 100

funcs = ['relu', 'sigmoid', 'identity']
error = ['mse', 'bce']
results = []
for i in range(NUM_TESTS):
    for f,g,loss in it.product(*[funcs,funcs,error]):
        results.append(test(f,g,loss))

columns = (['non-linear', 'w1', 'b1', 'w2', 'b2', 'loss'])
df = pd.DataFrame(results,columns=columns)
with pd.option_context('display.max_columns', None,
                       'display.max_rows', None,
                       'float_format', '{:,.2E}'.format,
                       'display.width', 240):
    print(f"running {NUM_TESTS} trials:")
    print(df.groupby(['non-linear'], as_index=False)
          .agg({'w1' : ['mean','std'], 'b1' : ['mean','std'], 'w2': ['mean','std'], 'b2': ['mean','std'], 'loss': ['mean','std']}))
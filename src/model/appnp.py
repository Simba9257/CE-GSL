import torch
from dgl.nn.pytorch import APPNPConv

import torch.nn as nn
import torch.nn.functional as F


class APPNP(nn.Module):
    def __init__(
        self,
        g,
        in_feats,
        hiddens,
        out_feats,
        activation,
        feat_drop,
        edge_drop,
        alpha,
        k,
    ):
        super(APPNP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, hiddens[0]))
        # hidden layers
        for i in range(1, len(hiddens)):
            self.layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        #self.layers.append(nn.Linear(hiddens[-1], out_feats))
        self.layers.append(nn.Linear(hiddens[-1], hiddens[-1], "mean"))
        self.linear0 = nn.Linear(in_feats, hiddens[-1])
        self.out = nn.Linear(2 * hiddens[-1], out_feats)

        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(self.g, h)
        return h

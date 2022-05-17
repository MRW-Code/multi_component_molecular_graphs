import os
import dgl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import dgl.nn.pytorch as dglnn

class GATNet_1(nn.Module):
    def __init__(self, n_feats):
        super(GATNet_1, self).__init__()
        self.embedding_size = 1024

        # Layers
        self.gat1A = dglnn.GATConv(n_feats, self.embedding_size, num_heads=3)
        self.linear1A = nn.Linear(self.embedding_size*3, self.embedding_size)
        self.output1A = nn.Linear(self.embedding_size, 1)
        self.relu = nn.LeakyReLU(True)
        self.pooling = dglnn.AvgPooling()

    def forward(self, bg, feats):
        x = self.gat1A(bg, feats)
        x = x.flatten(1)
        x = self.linear1A(x)
        x = self.relu(x)
        x = self.pooling(bg, x)
        x = self.output1A(x)
        return x.double()





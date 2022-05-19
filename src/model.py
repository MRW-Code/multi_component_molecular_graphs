import os
import dgl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import dgl.nn.pytorch as dglnn

class GATNet_1(nn.Module):
    def __init__(self, n_feats, embedding_size):
        super(GATNet_1, self).__init__()
        self.embedding_size = embedding_size

        # Layers
        self.gat1A = dglnn.GATConv(n_feats, self.embedding_size, num_heads=3, feat_drop=0.3)
        self.linear1A = nn.Linear(self.embedding_size*3, self.embedding_size)
        self.output1A = nn.Linear(self.embedding_size, 1)
        self.relu = nn.LeakyReLU(True)
        self.pooling = dglnn.AvgPooling()

        self.gatn = dglnn.GATConv(self.embedding_size, self.embedding_size, num_heads=3)

    def forward(self, bg, feats):
        x = self.gat1A(bg, feats)
        x = x.flatten(1)
        x = self.linear1A(x)

        for i in range(2):
            x = self.gatn(bg, x)
            x = x.flatten(1)
            x = self.linear1A(x)
            x = self.relu(x)

        x = self.pooling(bg, x)
        return x
        # x = self.output1A(x)
        # return x.double()


class DoubleNet(nn.Module):

    def __init__(self, n_feats, emb_size):
        super(DoubleNet, self).__init__()
        self.gat1 = GATNet_1(n_feats, emb_size)
        self.gat2 = GATNet_1(n_feats, emb_size)
        self.pooling = dglnn.AvgPooling()
        self.output = nn.Linear(emb_size*2, 1)

    def forward(self, bgA, featsA, bgB, featsB):
        x = self.gat1(bgA, featsA)
        y = self.gat2(bgB, featsB)

        cat_feats = torch.cat([x, y], axis=1)
        # cat_graphs = dgl.batch([bgA, bgB])
        # cat_feats = self.pooling(cat_graphs, cat_feats)
        z = self.output(cat_feats)
        return z.double()



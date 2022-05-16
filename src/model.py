import os

import dgl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import dgl.nn.pytorch as dglnn


class GCN(pl.LightningModule):
    def __init__(self, in_featA, in_featB, n_hidden):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_featA, n_hidden)
        self.conv2 = dglnn.GraphConv(in_featB, n_hidden)

        self.flatten = nn.Linear(20, 20)

        self.final = nn.Linear(n_hidden*2, 1)
        self.n_hidden = n_hidden

    def training_step(self, batch, batch_idx):
        graphA, graphB, label = batch
        graphA = dgl.add_self_loop(graphA)
        graphB = dgl.add_self_loop(graphB)

        feat_vec_a = torch.zeros([graphA.num_nodes(), self.n_hidden], dtype=torch.float)
        feat_vec_b = torch.zeros([graphB.num_nodes(), self.n_hidden], dtype=torch.float)

        h = self.conv1(graphA, feat_vec_a)
        i = self.conv2(graphB, feat_vec_b)

        # this is wrong time for multihead attention!!!
        j = torch.concat((h, h),axis=1)
        pred = self.final(j).mean().double()


        loss = F.mse_loss(pred, label)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer


class GATNet_1(pl.LightningModule):
    def __init__(self, n_feats):
        super(GATNet, self).__init__()

        self.embedding_size = 1024

        # Layers
        self.gat1A = dglnn.GATConv(n_feats, self.embedding_size, num_heads=1)
        self.linear1A = nn.Linear(self.embedding_size, self.embedding_size)
        self.output1A = nn.Linear(self.embedding_size, 1)

        # self.gat1B = dglnn.GATConv(n_feats, embedding_size, 1)
        # self.linear1A = nn.Linear(embedding_size * n_feats, embedding_size)
        # self.output1B = nn.Linear(embedding_size, 1)

    def forward(self, x):
        feat = torch.ones(x.num_nodes(), self.embedding_size)

        x = self.gat1A(x, feat)
        x = self.linear1A(x)
        x = self.output1A(x)
        return x


    def training_step(self, batch, batch_idx):
        graphA, graphB, label = batch
        graphA = dgl.add_self_loop(graphA)
        graphB = dgl.add_self_loop(graphB)

        pred = self.forward(graphA)


        loss = F.mse_loss(pred, label)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Classifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, 1)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = self.classify(h)
        # with g.local_scope():
        #     g.ndata['h'] = h
        #     # Calculate graph representation by average readout.
        #     hg = dgl.mean_nodes(g, 'h')
        #     return self.classify(hg)
        return h

class GATNet_1(nn.Module):
    def __init__(self, n_feats):
        super(GATNet_1, self).__init__()

        self.embedding_size = 1024

        # Layers
        self.gat1A = dglnn.GATConv(n_feats, self.embedding_size, num_heads=3)
        self.linear1A = nn.Linear(self.embedding_size*3, self.embedding_size)
        self.output1A = nn.Linear(self.embedding_size, n_feats)

        # self.gat1B = dglnn.GATConv(n_feats, embedding_size, 1)
        # self.linear1A = nn.Linear(embedding_size * n_feats, embedding_size)
        # self.output1B = nn.Linear(embedding_size, 1)

    def forward(self, bg, feats):
        x = self.gat1A(bg, feats)
        x = x.flatten(1)
        x = self.linear1A(x)
        x = self.output1A(x)
        return x





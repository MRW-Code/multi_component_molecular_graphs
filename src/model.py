import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import dgl.nn.pytorch as dglnn

class GCN(pl.LightningModule):
    def __init__(self, in_featA, in_featB, n_hidden):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_featA, n_hidden, activation='relu')
        self.conv2 = dglnn.GraphConv(in_featB, n_hidden, activation='relu')
        self.final = nn.Linear(n_hidden*2, 1)
        self.n_hidden = n_hidden

    def training_step(self, batch, batch_idx):
        graphA, graphB, label = batch

        feat_vec_a = torch.zeros([graphA.num_nodes(), self.n_hidden], dtype=torch.float)
        feat_vec_b = torch.zeros([graphB.num_nodes(), self.n_hidden], dtype=torch.float)

        h = self.conv1(graphA, feat_vec_a)
        i = self.conv2(graphB, feat_vec_b)


        j = torch.concat([h, i],axis=1)
        pred = self.final(j)

        loss = F.mse_loss(pred, label)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer


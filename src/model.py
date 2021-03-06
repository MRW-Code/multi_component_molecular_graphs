import os
import dgl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import WeightAndSum
import numpy as np


class InputInitializer(nn.Module):
    """Initializde edge representations based on input node and edge features

    Parameters
    ----------
    in_node_feats : int
        Number of input node features
    in_edge_feats : int
        Number of input edge features
    """
    def __init__(self, in_node_feats, in_edge_feats):
        super(InputInitializer, self).__init__()

        self.project_nodes = nn.Linear(in_node_feats, in_node_feats)
        self.project_edges = nn.Linear(in_edge_feats, in_edge_feats)

    def forward(self, bg, node_feats, edge_feats):
        """Initialize input representations.

        Project the node/edge features and then concatenate the edge representations with the
        representations of their source nodes.
        """
        node_feats = self.project_nodes(node_feats)
        edge_feats = self.project_edges(edge_feats)

        bg = bg.local_var()
        bg.ndata['hv'] = node_feats
        bg.apply_edges(dgl.function.copy_u('hv', 'he'))
        return torch.cat([bg.edata['he'], edge_feats], dim=1)

class EdgeGraphConv(nn.Module):
    """Apply graph convolution over an input edge signal.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
    """
    def __init__(self, in_feats, out_feats, activation=F.relu):
        super(EdgeGraphConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, graph, feat):
        """Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input edge features.

        Returns
        -------
        torch.Tensor
            The output features.
        """
        graph = graph.local_var()

        if self.in_feats > self.out_feats:
            # multiply by W first to reduce the feature size for aggregation.
            feat = self.linear(feat)
            graph.edata['h'] = feat
            graph.update_all(dgl.function.copy_e('h', 'm'), dgl.function.sum('m', 'h'))
            rst = graph.ndata['h']
        else:
            # aggregate first then multiply by W
            graph.edata['h'] = feat
            graph.update_all(dgl.function.copy_e('h', 'm'), dgl.function.sum('m', 'h'))
            rst = graph.ndata['h']
            rst = self.linear(rst)

        if self.activation is not None:
            rst = self.activation(rst)
        return rst

class WeightedSumAndMax(nn.Module):
    r"""Apply weighted sum and max pooling to the node
    representations and concatenate the results.

    Parameters
    ----------
    in_feats : int
        Input node feature size
    """
    def __init__(self, in_feats):
        super(WeightedSumAndMax, self).__init__()

        self.weight_and_sum = WeightAndSum(in_feats)

    def forward(self, bg, feats):
        """Readout

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization

        Returns
        -------
        h_g : FloatTensor of shape (B, 2 * M1)
            * B is the number of graphs in the batch
            * M1 is the input node feature size, which must match
              in_feats in initialization
        """
        h_g_sum = self.weight_and_sum(bg, feats)
        with bg.local_scope():
            bg.ndata['h'] = feats
            h_g_max = dgl.max_nodes(bg, 'h')
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        return h_g

class DNB(nn.Module):

    def __init__(self, n_feats, e_feats, emb_size, num_heads):
        super(DNB, self).__init__()
        self.name = 'DNB'
        self.join_feats1A = InputInitializer(n_feats, e_feats)
        self.join_feats2A = EdgeGraphConv(n_feats+e_feats, n_feats+e_feats)
        self.gat1A = dglnn.GATConv(n_feats + e_feats, emb_size, num_heads=num_heads,
                                   attn_drop=0, feat_drop=0, activation=nn.LeakyReLU())
        self.linear1A = nn.Sequential(nn.Linear(emb_size*num_heads, emb_size),
                                      nn.Dropout(0.1))
        self.linear2A = nn.Sequential(nn.Linear(emb_size*2, emb_size),
                                      nn.Dropout(0.1))

        self.join_feats1B = InputInitializer(n_feats, e_feats)
        self.join_feats2B = EdgeGraphConv(n_feats + e_feats, n_feats + e_feats)
        self.gat1B = dglnn.GATConv(n_feats+e_feats, emb_size, num_heads=num_heads,
                                   attn_drop=0, feat_drop=0, activation=nn.LeakyReLU())
        self.linear1B = nn.Sequential(nn.Linear(emb_size*num_heads, emb_size),
                                      nn.Dropout(0.1))
        self.linear2B = nn.Sequential(nn.Linear(emb_size*2, emb_size),
                                      nn.Dropout(0.1))


        self.flatten = nn.Flatten(1)
        self.pool = WeightedSumAndMax(emb_size)

        self.output = nn.Sequential(nn.Linear(emb_size*2, emb_size),
                                    nn.LeakyReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(emb_size, 1))
        self.sig = nn.Sigmoid()

    def forward(self, bgA, bgB):

        # FOR GRAPH A
        atom_featsA = bgA.ndata['atomic'].double()
        bond_featsA = bgA.edata['bond'].double()
        x = self.join_feats1A(bgA, atom_featsA, bond_featsA)
        x = self.join_feats2A(bgA, x)
        x = self.gat1A(bgA, x)
        x = self.flatten(x)
        x = self.linear1A(x)
        x = self.pool(bgA, x)
        x = self.linear2A(x)

        # FOR GRAPH B
        atom_featsB = bgB.ndata['atomic'].double()
        bond_featsB = bgB.edata['bond'].double()
        y = self.join_feats1B(bgB, atom_featsB, bond_featsB)
        y = self.join_feats2B(bgB, y)
        y = self.gat1B(bgB, y)
        y = self.flatten(y)
        y = self.linear1B(y)
        y = self.pool(bgB, y)
        y = self.linear2B(y)

        # Cat
        z = torch.cat([x, y], axis=1)
        z = self.output(z)
        return z.double()
        # return self.sig(z)

class DNBDeep(nn.Module):

    def __init__(self, n_feats, e_feats, emb_size, num_heads):
        super(DNBDeep, self).__init__()
        self.name = 'DNBDeep'
        # For molecules
        self.join_features1A = InputInitializer(n_feats, e_feats)
        self.join_features2A = EdgeGraphConv(n_feats + e_feats, n_feats + e_feats)
        self.gat1A = dglnn.GATConv(n_feats + e_feats, emb_size, num_heads=num_heads,
                                                 attn_drop=0, feat_drop=0, activation=nn.LeakyReLU())
        self.linear1A = nn.Sequential(nn.Flatten(1),
                        nn.Linear(emb_size * num_heads, emb_size),
                        nn.Dropout(0.1))

        self.gat2A = dglnn.GATConv(emb_size, emb_size, num_heads=num_heads,
                                                 attn_drop=0, feat_drop=0, activation=nn.LeakyReLU())
        self.linear2A = nn.Sequential(nn.Flatten(1),
                        nn.Linear(emb_size * num_heads, emb_size),
                        nn.Dropout(0.1))
        self.gat3A = dglnn.GATConv(emb_size, emb_size, num_heads=num_heads,
                                   attn_drop=0, feat_drop=0, activation=nn.LeakyReLU())
        self.linear3A = nn.Sequential(nn.Flatten(1),
                                      nn.Linear(emb_size * num_heads, emb_size),
                                      nn.Dropout(0.1))


        self.graph_poolA = WeightedSumAndMax(emb_size)
        self.pool_linearA = nn.Sequential(nn.Linear(emb_size * 2, emb_size),
                                         nn.Dropout(0.1))

        # For solvents
        self.join_features1B = InputInitializer(n_feats, e_feats)
        self.join_features2B = EdgeGraphConv(n_feats + e_feats, n_feats + e_feats)
        self.gat1B = dglnn.GATConv(n_feats + e_feats, emb_size, num_heads=num_heads,
                                                 attn_drop=0, feat_drop=0, activation=nn.LeakyReLU())
        self.linear1B = nn.Sequential(nn.Flatten(1),
                        nn.Linear(emb_size * num_heads, emb_size),
                        nn.Dropout(0.1))
        self.gat2B = dglnn.GATConv(emb_size, emb_size, num_heads=num_heads,
                                                 attn_drop=0, feat_drop=0, activation=nn.LeakyReLU())
        self.linear2B = nn.Sequential(nn.Flatten(1),
                        nn.Linear(emb_size * num_heads, emb_size),
                        nn.Dropout(0.1))
        self.gat3B = dglnn.GATConv(emb_size, emb_size, num_heads=num_heads,
                                   attn_drop=0, feat_drop=0, activation=nn.LeakyReLU())
        self.linear3B = nn.Sequential(nn.Flatten(1),
                                      nn.Linear(emb_size * num_heads, emb_size),
                                      nn.Dropout(0.1))

        self.graph_poolB = WeightedSumAndMax(emb_size)
        self.pool_linearB = nn.Sequential(nn.Linear(emb_size * 2, emb_size),
                                         nn.Dropout(0.1))

        # For joined / both molecule and solvent details

        self.output = nn.Sequential(nn.Linear(emb_size*2, emb_size),
                                    nn.LeakyReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(emb_size, 1))


    def forward(self, bgA, bgB):
        #################
        # FOR molecules #
        #################
        atom_featsA = bgA.ndata['atomic'].double()
        bond_featsA = bgA.edata['bond'].double()
        x = self.join_features1A(bgA, atom_featsA, bond_featsA)
        x = self.join_features2A(bgA, x)
        x = self.gat1A(bgA, x)
        x = self.linear1A(x)

        # Add a second GAT layer
        x = self.gat2A(bgA, x)
        x = self.linear2A(x)

        # Add a 3rd GAT layer
        # x = self.gat3A(bgA, x)
        # x = self.linear3A(x)

        # Prep molecules for concat
        x = self.graph_poolA(bgA, x)
        x = self.pool_linearA(x)

        #################
        # FOR Solvents #
        #################
        atom_featsB = bgB.ndata['atomic'].double()
        bond_featsB = bgB.edata['bond'].double()
        y = self.join_features1B(bgB, atom_featsB, bond_featsB)
        y = self.join_features2B(bgB, y)
        y = self.gat1B(bgB, y)
        y = self.linear1B(y)

        # Add a second GAT layer
        y = self.gat2B(bgB, y)
        y = self.linear2B(y)

        # Add a 3rd GAT layer
        # y = self.gat3B(bgB, y)
        # y = self.linear3B(y)

        # Prep molecules for concat
        y = self.graph_poolB(bgB, y)
        y = self.pool_linearB(y)

        ##########
        # Concat #
        ##########
        z = torch.cat([x, y], axis=1)
        z = self.output(z)
        return z.double()

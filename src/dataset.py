import torch
from torch.utils.data import Dataset
from src.processing import get_useful_columns
import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit import Chem
from dgllife.utils import smiles_to_bigraph
import pytorch_lightning as pl
import dgl
from torch_geometric.utils import from_networkx

class MultiCompSolDataset(Dataset):
    def __init__(self, transforms=None, target_transforms=None):
        self.df = get_useful_columns('./data/dataset_vas_et_al.csv')
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return self.df.shape[0]

    def featurize_atoms(self, mol):
        feats = np.empty(len(mol.GetAtoms()))
        for idx, atom in enumerate(mol.GetAtoms()):
            feats[idx] = atom.GetAtomicNum()
        return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}

    def featurize_bonds(self, mol):
        feats = []
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        for bond in mol.GetBonds():
            btype = bond_types.index(bond.GetBondType())
            # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
            feats.extend([btype, btype])
        return {'type': torch.tensor(feats).reshape(-1, 1).float()}

    def __getitem__(self, idx):
        smilesA, smilesB, label = self.df.loc[idx, :].values
        graphA = smiles_to_bigraph(smilesA, node_featurizer=self.featurize_atoms,
                              edge_featurizer = self.featurize_bonds)
        graphA = dgl.add_self_loop(graphA)
        # graphA = dgl.to_networkx(graphA)
        # graphA = from_networkx(graphA)


        graphB = smiles_to_bigraph(smilesB, node_featurizer=self.featurize_atoms,
                                   edge_featurizer=self.featurize_bonds)
        graphB = dgl.add_self_loop(graphB)
        # graphB = dgl.to_networkx(graphB)
        # graphB = from_networkx(graphB)

        return graphA, graphB, label

class MultiCompSolDatasetv2(dgl.data.DGLDataset):
    def __init__(self):
        super().__init__(name='multicompsoldatasetv2')


    def featurize_atoms(self, mol):
        feats = np.empty(len(mol.GetAtoms()))
        for idx, atom in enumerate(mol.GetAtoms()):
            feats[idx] = atom.GetAtomicNum()
        return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}

    def featurize_bonds(self, mol):
        feats = []
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        for bond in mol.GetBonds():
            btype = bond_types.index(bond.GetBondType())
            # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
            feats.extend([btype, btype])
        return {'type': torch.tensor(feats).reshape(-1, 1).float()}

    def process(self):
        self.graphsA = []
        self.graphsB = []
        self.labels = []
        df = get_useful_columns('./data/dataset_vas_et_al.csv')
        for idx in range(len(df)):
            smilesA, smilesB, label = df.loc[idx, :].values

            # Make list of graph A
            graphA = smiles_to_bigraph(smilesA, node_featurizer=self.featurize_atoms,
                                       edge_featurizer=self.featurize_bonds)
            graphA = dgl.add_self_loop(graphA)
            self.graphsA.append(graphA)

            # Make list of graph B
            graphB = smiles_to_bigraph(smilesB, node_featurizer=self.featurize_atoms,
                                       edge_featurizer=self.featurize_bonds)
            graphB = dgl.add_self_loop(graphB)
            self.graphsB.append(graphB)

            # Get labels
            self.labels.append(label)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphsA[i], self.graphsB[i], self.labels[i]

    def __len__(self):
        if len(self.graphsA) == len(self.graphsB):
            return len(self.graphsA)
        else:
            raise AttributeError('Number of graphs not the same for component A and B')

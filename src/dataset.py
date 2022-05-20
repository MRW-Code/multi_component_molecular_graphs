import pandas as pd

from src.utils import args, device
import torch
from torch.utils.data import Dataset
from src.processing import get_useful_columns
import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit import Chem
from dgllife.utils import smiles_to_bigraph
import pytorch_lightning as pl
from dgllife.utils.featurizers import AttentiveFPBondFeaturizer, ConcatFeaturizer, BaseAtomFeaturizer
import dgl
from dgllife.utils import *

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
        df = get_useful_columns('./data/solubility/dataset_vas_et_al.csv')
        for idx in range(len(df)):
            smilesA, smilesB, label = df.loc[idx, :].values

            # Make list of graph A
            graphA = smiles_to_bigraph(smilesA, node_featurizer=self.featurize_atoms,
                                       edge_featurizer=self.featurize_bonds)
            graphA = dgl.add_self_loop(graphA)
            self.graphsA.append(graphA.to(device))

            # Make list of graph B
            graphB = smiles_to_bigraph(smilesB, node_featurizer=self.featurize_atoms,
                                       edge_featurizer=self.featurize_bonds)
            # graphB = dgl.add_self_loop(graphB)
            self.graphsB.append(graphB.to(device))

            # Get labels
            self.labels.append(label)
        self.labels = torch.tensor(self.labels, device=device)

    def __getitem__(self, i):
        return self.graphsA[i], self.graphsB[i], self.labels[i]

    def __len__(self):
        if len(self.graphsA) == len(self.graphsB):
            return len(self.graphsA)
        else:
            raise AttributeError('Number of graphs not the same for component A and B')

class MultiCompSolDatasetv3(dgl.data.DGLDataset):

    def __init__(self, use_one_hot):
        self.use_one_hot = use_one_hot
        super().__init__(name='multicompsoldatasetv3')


    def featurize_atoms(self, mol):
        if not self.use_one_hot:
            atom_feats = [atomic_number,
               atom_degree,
               atom_total_degree,
               atom_explicit_valence,
               atom_implicit_valence,
               atom_total_num_H,
               atom_formal_charge,
               atom_num_radical_electrons,
               atom_is_aromatic,
               atom_is_in_ring,
               atom_mass,
               atom_is_chiral_center]
        else:
            atom_feats = [atom_type_one_hot,
                          atomic_number_one_hot,
                          atom_degree_one_hot,
                          atom_total_degree_one_hot,
                          atom_explicit_valence_one_hot,
                          atom_implicit_valence_one_hot,
                          atom_hybridization_one_hot,
                          atom_total_num_H_one_hot,
                          atom_formal_charge_one_hot,
                          atom_num_radical_electrons_one_hot,
                          atom_is_aromatic_one_hot,
                          atom_is_in_ring_one_hot,
                          atom_chiral_tag_one_hot,
                          atom_chirality_type_one_hot]

        atom_featurizer = ConcatFeaturizer(atom_feats)
        mol_atom_featurizer = BaseAtomFeaturizer({'atomic': atom_featurizer})
        return mol_atom_featurizer(mol)

    def featurize_bonds(self, mol):
        bond_feat_maker = AttentiveFPBondFeaturizer('bond', self_loop=False)
        bond_feats = bond_feat_maker(mol)
        try:
            if bond_feats['bond'].shape[1] < 10:
                print(mol)
        except:
            print(f'Error with {Chem.MolToSmiles(mol)} - maybe no bonds!')
            print('wait')

        return bond_feats

    def process(self):
        self.graphsA = []
        self.graphsB = []
        self.labels = []
        df = get_useful_columns('./data/solubility/dataset_vas_et_al.csv')
        for idx in range(len(df)):
            smilesA, smilesB, label = df.loc[idx, :].values

            # Make list of graph A
            graphA = smiles_to_bigraph(smilesA, node_featurizer=self.featurize_atoms,
                                       edge_featurizer=self.featurize_bonds,
                                       explicit_hydrogens=True)
            graphA = dgl.add_self_loop(graphA)
            self.graphsA.append(graphA.to(device))

            # Make list of graph B
            graphB = smiles_to_bigraph(smilesB, node_featurizer=self.featurize_atoms,
                                       edge_featurizer=self.featurize_bonds,
                                       explicit_hydrogens=True)
            graphB = dgl.add_self_loop(graphB)
            self.graphsB.append(graphB.to(device))

            # Get labels
            self.labels.append(label)
        self.labels = torch.tensor(self.labels, device=device)

    def __getitem__(self, i):
        return self.graphsA[i], self.graphsB[i], self.labels[i]

    def __len__(self):
        if len(self.graphsA) == len(self.graphsB):
            return len(self.graphsA)
        else:
            raise AttributeError('Number of graphs not the same for component A and B')

class CCDataset(dgl.data.DGLDataset):

    def __init__(self, use_one_hot):
        self.use_one_hot = use_one_hot
        super().__init__(name='ccdataset')

    def featurize_atoms(self, mol):
        if not self.use_one_hot:
            atom_feats = [atomic_number,
               atom_degree,
               atom_total_degree,
               atom_explicit_valence,
               atom_implicit_valence,
               atom_total_num_H,
               atom_formal_charge,
               atom_num_radical_electrons,
               atom_is_aromatic,
               atom_is_in_ring,
               atom_mass,
               atom_is_chiral_center]
        else:
            atom_feats = [atom_type_one_hot,
                          atomic_number_one_hot,
                          atom_degree_one_hot,
                          atom_total_degree_one_hot,
                          atom_explicit_valence_one_hot,
                          atom_implicit_valence_one_hot,
                          atom_hybridization_one_hot,
                          atom_total_num_H_one_hot,
                          atom_formal_charge_one_hot,
                          atom_num_radical_electrons_one_hot,
                          atom_is_aromatic_one_hot,
                          atom_is_in_ring_one_hot,
                          atom_chiral_tag_one_hot,
                          atom_chirality_type_one_hot]

        atom_featurizer = ConcatFeaturizer(atom_feats)
        mol_atom_featurizer = BaseAtomFeaturizer({'atomic': atom_featurizer})
        return mol_atom_featurizer(mol)

    def featurize_bonds(self, mol):
        bond_feat_maker = AttentiveFPBondFeaturizer('bond', self_loop=False)
        bond_feats = bond_feat_maker(mol)
        try:
            if bond_feats['bond'].shape[1] < 10:
                print(mol)
        except:
            print(f'Error with {Chem.MolToSmiles(mol)} - maybe no bonds!')
            print('wait')

        return bond_feats

    def process(self):
        self.graphsA = []
        self.graphsB = []
        self.labels = []
        smilesA = pd.read_csv('./data/cocrystal/component1_smiles.csv', index_col=0)
        smilesB = pd.read_csv('./data/cocrystal/component2_smiles.csv', index_col=0)
        raw_data = pd.read_csv('./data/cocrystal/jan_raw_data.csv')
        df = pd.merge(smilesA, raw_data, left_on='api', right_on='Component1')\
            .drop(['api', 'Component1'], axis=1)
        df = pd.merge(smilesB, df, left_on='api', right_on='Component2')\
            .drop(['api', 'Component2'], axis=1)
        for idx in range(len(df)):
            smilesA, smilesB, label = df.loc[idx, :].values

            # Make list of graph A
            graphA = smiles_to_bigraph(smilesA, node_featurizer=self.featurize_atoms,
                                       edge_featurizer=self.featurize_bonds,
                                       explicit_hydrogens=True)
            graphA = dgl.add_self_loop(graphA)
            self.graphsA.append(graphA.to(device))

            # Make list of graph B
            graphB = smiles_to_bigraph(smilesB, node_featurizer=self.featurize_atoms,
                                       edge_featurizer=self.featurize_bonds,
                                       explicit_hydrogens=True)
            graphB = dgl.add_self_loop(graphB)
            self.graphsB.append(graphB.to(device))

            # Get labels
            self.labels.append(label)
        self.labels = torch.tensor(self.labels, device=device)

    def __getitem__(self, i):
        return self.graphsA[i], self.graphsB[i], self.labels[i]

    def __len__(self):
        if len(self.graphsA) == len(self.graphsB):
            return len(self.graphsA)
        else:
            raise AttributeError('Number of graphs not the same for component A and B')
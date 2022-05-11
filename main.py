from src.utils import args
from src.dataset import MultiCompSolDataset
from dgl.dataloading import GraphDataLoader
from src.model import GCN
import pytorch_lightning as pl

if __name__ == '__main__':
    dataloader = GraphDataLoader(MultiCompSolDataset(), batch_size=64, shuffle=True)

    test = next(iter(dataloader))
    print('done')


    model = GCN(in_featA=100, in_featB=100, n_hidden=10)
    trainer = pl.Trainer()
    trainer.fit(model, dataloader)


    # train_featuresA, train_featuresB, train_labels = next(iter(dls))
    # Convert the SMILES codes into molcular graphs


    # Define and create the model


    # Pytorch training loop


    # Return some metics and or figures


    print('done')
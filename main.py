from src.utils import args
from src.dataset import MultiCompSolDataset , MultiCompSolDatasetv2
from dgl.dataloading import GraphDataLoader
from src.model import *
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from dgl.data import DGLDataset
from dgl.dataloading import batch_graphs

if __name__ == '__main__':
    dataset = MultiCompSolDatasetv2()
    dataloader = GraphDataLoader(dataset, batch_size=10, shuffle=False)

    model = GATNet_1(1)
    opt = torch.optim.Adam(model.parameters())

    epochs = 10


    for epoch in range(epochs):
        running_loss = 0.0

        for batch_id, batch_data in enumerate(dataloader):
            batched_graphA, batched_graphB, labels = batch_data

            feats = batched_graphA.ndata['atomic']
            logits = model(batched_graphA, feats)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            print(f'[{epoch + 1}, {epoch + 1:5d}] loss: {running_loss / 1:.3f}')
            running_loss = 0.0



        print('done')

    print('done')


    # model = GCN(in_featA=20, in_featB=20, n_hidden=20)
    # model = GATNet(test[0].num_nodes())
    # trainer = pl.Trainer(max_epochs=10, min_epochs=10, log_every_n_steps=2)
    # trainer.fit(model, dataloader)


    # train_featuresA, train_featuresB, train_labels = next(iter(dls))
    # Convert the SMILES codes into molcular graphs


    # Define and create the model


    # Pytorch training loop


    # Return some metics and or figures


    print('done')
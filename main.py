from src.utils import args, device
from src.dataset import MultiCompSolDatasetv2
from dgl.dataloading import GraphDataLoader
from src.model import GATNet_1
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from dgl.data import DGLDataset
from dgl.dataloading import batch_graphs
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

if __name__ == '__main__':
    print(device)
    dataset = MultiCompSolDatasetv2()
    dataloader = GraphDataLoader(dataset, batch_size=256, shuffle=False)

    model = GATNet_1(1)

    opt = torch.optim.Adam(model.parameters(), lr=0.00001)
    epochs = 500
    model.to(device)
    for epoch in range(epochs):
        ls = []
        for batch_id, batch_data in enumerate(dataloader):
            batched_graphA, batched_graphB, labels = batch_data
            labels = labels.reshape(-1, 1)
            feats = batched_graphA.ndata['atomic']

            logits = model(batched_graphA, feats)
            loss = F.mse_loss(logits, labels)
            ls.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            # metrics
            if torch.cuda.is_available():
                labels = labels.detach().cpu().numpy()
                logits = logits.detach().cpu().numpy()
            else:
                labels = labels.detach().numpy()
                logits = logits.detach().numpy()

            r2 = r2_score(labels, logits)
            mse = mean_squared_error(labels, logits)
            mae = mean_absolute_error(labels, logits)
            mape = mean_absolute_percentage_error(labels, logits)

        tqdm.write(f'Epoch {epoch:.3f}, Loss={np.mean(ls):.3f}, r2={r2:.3f}, mse={mse:.3f}, mae={mae:.3f}, mape={mape:.3f}')

            # running_loss += loss.item()
            # print(f'[{epoch + 1}, {epoch + 1:5d}] loss: {running_loss / 1:.3f}')
            # running_loss = 0.0


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
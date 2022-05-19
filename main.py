from src.utils import args, device
from src.dataset import MultiCompSolDatasetv2
from dgl.dataloading import GraphDataLoader
from src.model import GATNet_1, DoubleNet
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from dgl.data import DGLDataset
from dgl.dataloading import batch_graphs
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from dgllife.utils.splitters import SingleTaskStratifiedSplitter
import os


if __name__ == '__main__':
    os.makedirs('./checkpoints/models', exist_ok=True)

    # Load dataset
    dataset = MultiCompSolDatasetv2()

    # Split dataset into train/val/test
    # There is also a kfold function here which might be helpful at some point
    train_dataset, val_dataset, test_dataset = SingleTaskStratifiedSplitter.train_val_test_split(dataset, dataset.labels.reshape(-1, 1), 0)

    # Create dataloaders for each split
    train_dataloader = GraphDataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = GraphDataLoader(val_dataset, batch_size=64, shuffle=True)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=1, shuffle=True)

    # Get model and stuff for training
    model = DoubleNet(1, 512)
    # opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10, threshold=0.1,
                                  threshold_mode='rel',  verbose=True)
    criterion = torch.nn.MSELoss()
    epochs = 10000
    model.to(device)
    min_valid_loss = np.inf
    counter = 0
    # Training loop
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for batch_id, batch_data in enumerate(train_dataloader):
            # Get labels and features
            batched_graphA, batched_graphB, labels = batch_data
            labels = labels.reshape(-1, 1)
            featsA = batched_graphA.ndata['atomic']
            featsB = batched_graphB.ndata['atomic']

            # Get preds
            opt.zero_grad()
            logits = model(batched_graphA, featsA, batched_graphB, featsB)
            loss = criterion(logits, labels)

            # backwards pass
            loss.backward()
            opt.step()
            train_loss += loss.item()


        valid_loss = 0.0
        model.eval()
        for batch_id, batch_data in enumerate(val_dataloader):
            # Get labels and features
            batched_graphA, batched_graphB, labels = batch_data
            labels = labels.reshape(-1, 1)
            featsA = batched_graphA.ndata['atomic']
            featsB = batched_graphB.ndata['atomic']

            target = model(batched_graphA, featsA, batched_graphB, featsB)
            loss = criterion(target, labels)
            valid_loss += loss.item()

        print(f'Epoch {epoch + 1} \t Training Loss: {train_loss / len(train_dataloader)} \t Validation Loss: {valid_loss / len(val_dataloader)}')

        if min_valid_loss - valid_loss < 0.1:
            counter += 1
        else:
            counter = 0
        # print(min_valid_loss, valid_loss, min_valid_loss - valid_loss, counter)

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            # torch.save(model.state_dict(), './checkpoints/models/best_model.pth')
            torch.save(model, './checkpoints/models/best_model.pth')

        # Early stopping
        if counter >= 20:
            print('No imporvement in 10 epochs, early stopping triggered')
            break

        scheduler.step(valid_loss)


    model = torch.load('./checkpoints/models/best_model.pth')
    model.eval()
    all_preds = np.empty(len(test_dataloader))
    all_labels = np.empty(len(test_dataloader))
    for batch_id, batch_data in enumerate(test_dataloader):
        # Get labels and features
        batched_graphA, batched_graphB, labels = batch_data
        labels = labels.reshape(-1, 1)
        featsA = batched_graphA.ndata['atomic']
        featsB = batched_graphB.ndata['atomic']

        preds = model(batched_graphA, featsA, batched_graphB, featsB)

        # metrics
        if torch.cuda.is_available():
            all_labels[batch_id] = labels.detach().cpu().numpy()
            all_preds[batch_id] = preds.detach().cpu().numpy()
        else:
            all_labels[batch_id] = labels.detach().numpy()
            all_preds[batch_id] = preds.detach().numpy()

    r2 = r2_score(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    mape = mean_absolute_percentage_error(all_labels, all_preds)
    print('External test set:')
    print(f' r2={r2:.3f}, mse={mse:.3f}, mae={mae:.3f}, mape={mape:.3f}')

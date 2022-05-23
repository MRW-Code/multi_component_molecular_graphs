from src.utils import args, device
from src.dataset import *
from dgl.dataloading import GraphDataLoader
from src.model import *
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from dgl.data import DGLDataset
from dgl.dataloading import batch_graphs
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,\
    mean_absolute_percentage_error, accuracy_score
from dgllife.utils.splitters import SingleTaskStratifiedSplitter
import os
import mlflow


if __name__ == '__main__':
    emb_size = 4096
    num_heads = 3     # 6
    lr = 1e-4           # usual 1e-3
    bs = 32
    os.makedirs('./checkpoints/models', exist_ok=True)

    # Load dataset
    dataset = MultiCompSolDatasetv3(use_one_hot=True)
    # dataset = CCDataset(use_one_hot=True)

    # Split dataset into train/val/test
    # There is also a kfold function here which might be helpful at some point
    train_dataset, val_dataset, test_dataset = \
        SingleTaskStratifiedSplitter.train_val_test_split(dataset,
                                                          dataset.labels.reshape(-1, 1),
                                                          0,
                                                          frac_train=0.8,
                                                          frac_val=0.1,
                                                          frac_test=0.1,
                                                          random_state=0)

    # Create dataloaders for each split
    train_dataloader = GraphDataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_dataloader = GraphDataLoader(val_dataset, batch_size=bs, shuffle=True)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=1, shuffle=True)

    n_feats = dataset.graphsA[0].ndata['atomic'].shape[1]
    e_feats = dataset.graphsA[0].edata['bond'].shape[1]

    # Get model
    model = DoubleNetBoth(n_feats=n_feats, e_feats=e_feats, emb_size=emb_size, num_heads=3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # opt = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10, threshold=0.1,
                                  threshold_mode='rel',  verbose=True)
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.BCELoss()

    epochs = 10
    model.to(device)
    min_valid_loss = np.inf
    counter = 0
    min_rmse = 0.0

    with mlflow.start_run() as run:
        mlflow.log_params({'emb_size': emb_size,
                           'num_heads': num_heads,
                           'lr': lr,
                           'bs': bs})

        # Training loop
        for epoch in range(epochs):
            train_loss = 0.0
            model.train()
            for batch_id, batch_data in enumerate(train_dataloader):
                # Get labels and features
                batched_graphA, batched_graphB, labels = batch_data
                labels = labels.unsqueeze(1)

                # Get preds
                opt.zero_grad()
                # logits = model(batched_graphA, featsA, batched_graphB, featsB)
                logits = model(batched_graphA, batched_graphB)
                loss = criterion(logits, labels)
                mlflow.log_metric('train_loss', value=float(loss), step=epoch)
                # backwards pass
                loss.backward()
                opt.step()
                train_loss += loss.item()


            valid_loss = 0.0
            model.eval()
            all_targets = np.empty(len(test_dataloader))
            all_labels = np.empty(len(test_dataloader))
            for batch_id, batch_data in enumerate(val_dataloader):
                # Get labels and features
                batched_graphA, batched_graphB, labels = batch_data
                labels = labels.unsqueeze(1)

                # target = model(batched_graphA, featsA, batched_graphB, featsB)
                target = model(batched_graphA, batched_graphB)
                loss = criterion(target.float(), labels.float())
                mlflow.log_metric('val_loss', value=float(loss), step=epoch)
                valid_loss += loss.item()

                # metrics
                if torch.cuda.is_available() and not args.cpu:
                    all_labels[batch_id] = labels.detach().cpu().numpy()[batch_id]
                    all_targets[batch_id] = target.detach().cpu().numpy()[batch_id]
                else:
                    all_labels[batch_id] = labels.detach().numpy()[batch_id]
                    all_targets[batch_id] = target.detach().numpy()[batch_id]

            r2 = r2_score(all_labels, all_targets)
            rmse = mean_squared_error(all_labels, all_targets, squared=False)
            mae = mean_absolute_error(all_labels, all_targets)
            mape = mean_absolute_percentage_error(all_labels, all_targets)

            print(f'Epoch {epoch + 1} Training Loss: {train_loss / len(train_dataloader):.3f} Validation Loss: ' +\
                  f'{valid_loss / len(val_dataloader):.3f} r2:{r2:.3f}, rmse:{rmse:.3f}, mape:{mape:.3f}')
            print(f'\t Validation Metrics: r2={r2:.3f}, rmse={rmse:.3f}, mae={mae:.3f}, mape={mape:.3f}')
            mlflow.log_metric('val_r2', value=float(r2), step=epoch)
            mlflow.log_metric('val_rmse', value=float(rmse), step=epoch)
            mlflow.log_metric('val_mae', value=float(mae), step=epoch)
            mlflow.log_metric('val_mape', value=float(mape), step=epoch)


            if min_valid_loss - valid_loss < 0.1:
                counter += 1
            else:
                counter = 0
            # print(min_valid_loss, valid_loss, min_valid_loss - valid_loss, counter)

            if min_valid_loss > valid_loss:
                min_valid_loss = valid_loss

            # if min_rmse > rmse or min_rmse == 0.0 and rmse > 0:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                # min_rmse = rmse
                # Saving State Dict
                # torch.save(model.state_dict(), './checkpoints/models/best_model.pth')
                torch.save(model, './checkpoints/models/best_model.pth')
                mlflow.pytorch.log_model(model, 'model')

            # Early stopping
            if counter >= 20:
                print('No improvement in 20 epochs, early stopping triggered')
                break

            scheduler.step(valid_loss)


    model = torch.load('./checkpoints/models/best_model.pth')
    model.eval()
    all_preds = np.empty(len(test_dataloader))
    all_labels = np.empty(len(test_dataloader))
    for batch_id, batch_data in enumerate(test_dataloader):
        # Get labels and features
        batched_graphA, batched_graphB, labels = batch_data
        labels = labels.unsqueeze(1)
        featsA = batched_graphA.ndata['atomic']
        featsB = batched_graphB.ndata['atomic']

        # preds = model(batched_graphA, featsA, batched_graphB, featsB)
        preds = model(batched_graphA, batched_graphB)

        # metrics
        if torch.cuda.is_available():
            all_labels[batch_id] = labels.detach().cpu().numpy()
            all_preds[batch_id] = preds.detach().cpu().numpy()
        else:
            all_labels[batch_id] = labels.detach().numpy()
            all_preds[batch_id] = preds.detach().numpy()

    r2 = r2_score(all_labels, all_preds)
    rmse = mean_squared_error(all_labels, all_preds, squared=False)
    mae = mean_absolute_error(all_labels, all_preds)
    mape = mean_absolute_percentage_error(all_labels, all_preds)
    print('External test set:')
    print(f' r2={r2:.3f}, rmse={rmse:.3f}, mae={mae:.3f}, mape={mape:.3f}')

    mlflow.log_metric('test_r2', value=float(r2), step=epoch)
    mlflow.log_metric('test_rmse', value=float(rmse), step=epoch)
    mlflow.log_metric('test_mae', value=float(mae), step=epoch)
    mlflow.log_metric('test_mape', value=float(mape), step=epoch)

    # acc = accuracy_score(all_labels, all_preds)
    # print('External test set:')
    # print(f' acc ={acc:.3f}')



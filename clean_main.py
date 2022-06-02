from src.utils import device
from src.dataset import *
from dgl.dataloading import GraphDataLoader

from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,\
    mean_absolute_percentage_error, accuracy_score
from dgllife.utils.splitters import SingleTaskStratifiedSplitter
import os
# import mlflow
import torch.nn as nn
import gc
from tabulate import tabulate
from natsort import natsorted
import matplotlib.pyplot as plt

def get_datsets(bs):
    os.makedirs('./checkpoints/models', exist_ok=True)
    dataset = MultiCompSolDatasetv3(use_one_hot=True)
    train_dataset, val_dataset, test_dataset = \
        SingleTaskStratifiedSplitter.train_val_test_split(dataset,
                                                          dataset.labels.reshape(-1, 1),
                                                          0,
                                                          frac_train=0.8,
                                                          frac_val=0.1,
                                                          frac_test=0.1,
                                                          random_state=0)
    train_dataloader = GraphDataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_dataloader = GraphDataLoader(val_dataset, batch_size=bs, shuffle=True)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=1, shuffle=True)

    n_feats = dataset.graphsA[0].ndata['atomic'].shape[1]
    e_feats = dataset.graphsA[0].edata['bond'].shape[1]

    data_dict = {'train': train_dataloader,
           'val': val_dataloader,
           'test': test_dataloader,
           'n_feats': n_feats,
           'e_feats': e_feats}
    return data_dict


def backprop(epoch, model, dataloader, optimizer, training):
    lf = nn.MSELoss()
    ls = []; preds = []; topreds = []
    for batch_id, batch_data in enumerate(dataloader):
        batched_graphA, batched_graphB, labels = batch_data
        labels = labels.unsqueeze(1)
        pred = model(batched_graphA, batched_graphB)
        loss = lf(pred, labels)
        ls.append(loss.item())
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        preds.append(pred.detach().cpu());
        topreds.append(labels.detach().cpu())
    if training: tqdm.write(f'Epoch {epoch},\tLoss = {np.mean(ls)}')
    return np.mean(ls) if training else (torch.cat(preds), torch.cat(topreds)), np.mean(ls)

def load_model(modelname, split_no, lr, n_feats, n_edges, emb_size, num_heads):
    import src.model
    model_class = getattr(src.model, modelname)
    model = model_class(n_feats, n_edges, emb_size, num_heads).double().to(device)
    optimizer = torch.optim.Adam(model.parameters() , lr=lr, weight_decay=1e-5)
    print(f"Creating new model: {model.name}")
    epoch = -1;
    accuracy_list = []
    return model, optimizer, epoch, accuracy_list

def save_model(model, split, optimizer, epoch, accuracy_list):
    folder = f'checkpoints/{args.model}_2_layers_8000_1024/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}model_{split}_{epoch}.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def prediction_scores(all_labels, all_targets):
    r2 = r2_score(all_labels, all_targets)
    rmse = mean_squared_error(all_labels, all_targets, squared=False)
    mae = mean_absolute_error(all_labels, all_targets)
    mape = mean_absolute_percentage_error(all_labels, all_targets)
    table = [['R2', r2], ['RMSE', rmse], ['mae', mae], ['mape', mape]]
    return table

if __name__ == '__main__':
    # HPARAMS
    N_SPLITS = 1
    EMB_SIZE = 1024
    NUM_HEADS = 3 # 6
    lr = 1e-4 # usual 1e-3
    bs = 16 #16

    data_dict = get_datsets(bs)

    num_epochs = 8000
    allpreds, alltopreds = [], []
    table = []; lf = nn.MSELoss(reduction = 'mean')
    for i in range(N_SPLITS):
        model, optimizer, epoch, accuracy_list = load_model(args.model, i, lr,
                                                            data_dict['n_feats'],
                                                            data_dict['e_feats'],
                                                            emb_size=EMB_SIZE,
                                                            num_heads=NUM_HEADS)
        # Training
        lowest_loss = np.inf
        val_acc = []
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1)), ncols=80):
            loss = backprop(e, model, data_dict['train'], optimizer, training=True)
            accuracy_list.append(loss[0])
            preds, val_loss = backprop(e, model, data_dict['val'], optimizer, training=False)
            val_acc.append(val_loss)

            if val_loss < lowest_loss:
                print('new_best_model, saving!')
                save_model(model, i, optimizer, e, accuracy_list)
                lowest_loss = val_loss
            if e == 4000:
                print(f'Time to reduce lr, current lr = {optimizer.param_groups[0]["lr"]}')
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]["lr"] / 10
                print(f'new lr = {optimizer.param_groups[0]["lr"]}')

            elif e == 6000:
                print(f'Time to reduce lr, current lr = {optimizer.param_groups[0]["lr"]}')
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
                print(f'new lr = {optimizer.param_groups[0]["lr"]}')

        # loss plot
        x = np.arange(0, num_epochs)
        plt.plot(x, accuracy_list, '-b', label='train_loss')
        plt.plot(x, val_acc, '-r', label='val_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='upper left')
        plt.savefig('loss_plot.png')
        plt.show()

        # Testing
        print('loading best model')
        model_paths = [f'./checkpoints/{args.model}_2_layers_8000_1024/{x}' for x in os.listdir(f'checkpoints/{args.model}_2_layers_8000_1024')]
        best_path = natsorted(model_paths)[-1]
        print(best_path)
        checkpoint = torch.load(best_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']

        preds, topreds = backprop(0, model, data_dict['val'], optimizer, training=False)[0]
        allpreds.append(preds); alltopreds.append(topreds)
        table.append([i, lf(preds, topreds).item()])

        ### Memory Cleaning
        model.cpu();
        gc.collect();
        torch.cuda.empty_cache()

    #RSME of Test Splits
    print(tabulate(table, headers=['Split', 'RMSE'], floatfmt=(None, '.4f')))
    table = np.array(table)
    avg, std = np.mean(table[:, 1]), np.std(table[:, 1])
    print(f'MSE = ' + "{:.4f}".format(avg) + u" \u00B1" + f' {"{:.4f}".format(std)}')
    # Scores
    allpreds, alltopreds = torch.cat(allpreds), torch.cat(alltopreds)
    print(tabulate(prediction_scores(alltopreds, allpreds), headers=['Metric', 'Value']))


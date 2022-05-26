from src.utils import device, args
from src.dataset import MultiCompSolDatasetv3
from dgl.dataloading import GraphDataLoader
from dgllife.utils.splitters import SingleTaskStratifiedSplitter
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,\
    mean_absolute_percentage_error, accuracy_score
import os
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from clean_main import load_model, get_datsets

if __name__ == '__main__':
    N_SPLITS = 1
    EMB_SIZE = 512
    NUM_HEADS = 3  # 6
    lr = 1e-4  # usual 1e-3
    bs = 1

    data_dict = get_datsets(bs)

    model_paths = sorted([f'./checkpoints/DNB_2000/{x}' for x in os.listdir('./checkpoints/DNB_2000')])
    # model_r2 = np.zeros(len(model_paths))
    # model_rmse = np.zeros(len(model_paths))
    # model_mae = np.zeros(len(model_paths))
    # model_mape = np.zeros(len(model_paths))
    # model_epoch = np.zeros(len(model_paths))
    # for idx, model_path in tqdm(enumerate(model_paths), nrows=40):
    #     epoch = re.findall(r'model_\d_(\d*).ckpt', model_path)[0]
    #     model, optimizer, epoch, accuracy_list = load_model(args.model, 0, lr,
    #                                                         data_dict['n_feats'],
    #                                                         data_dict['e_feats'],
    #                                                         emb_size=EMB_SIZE,
    #                                                         num_heads=NUM_HEADS)
    #     epoch = re.findall(r'model_\d_(\d*).ckpt', model_path)[0]
    #     checkpoint = torch.load(model_path, map_location=torch.device(device))
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     model.eval()
    #     all_preds = np.empty(len(data_dict['test']))
    #     all_labels = np.empty(len(data_dict['test']))
    #     for batch_id, batch_data in enumerate(data_dict['test']):
    #         # Get labels and features
    #         batched_graphA, batched_graphB, labels = batch_data
    #         labels = labels.unsqueeze(1)
    #         featsA = batched_graphA.ndata['atomic']
    #         featsB = batched_graphB.ndata['atomic']
    #
    #         # preds = model(batched_graphA, featsA, batched_graphB, featsB)
    #         preds = model(batched_graphA, batched_graphB)
    #
    #         # metrics
    #         if torch.cuda.is_available():
    #             all_labels[batch_id] = labels.detach().cpu().numpy()
    #             all_preds[batch_id] = preds.detach().cpu().numpy()
    #         else:
    #             all_labels[batch_id] = labels.detach().numpy()
    #             all_preds[batch_id] = preds.detach().numpy()
    #
    #     test_r2 = r2_score(all_labels, all_preds)
    #     test_rmse = mean_squared_error(all_labels, all_preds, squared=False)
    #     test_mae = mean_absolute_error(all_labels, all_preds)
    #     test_mape = mean_absolute_percentage_error(all_labels, all_preds)
    #
    #     model_r2[idx] = test_r2
    #     model_rmse[idx] = test_rmse
    #     model_mae[idx] = test_mae
    #     model_mape[idx] = test_mape
    #     model_epoch[idx] = int(epoch)
    #
    # fig, axs = plt.subplots(4, 1, constrained_layout=True)
    # axs[0].plot(model_epoch, model_r2, 'o')
    # axs[0].set_title('R2')
    # axs[0].set_xlabel('epoch')
    # axs[0].set_ylabel('R2')
    #
    # axs[1].plot(model_epoch, model_rmse, 'o')
    # axs[1].set_title('RMSE')
    # axs[1].set_xlabel('epoch')
    # axs[1].set_ylabel('RMSE')
    #
    # axs[2].plot(model_epoch, model_mae, 'o')
    # axs[2].set_title('MAE')
    # axs[2].set_xlabel('epoch')
    # axs[2].set_ylabel('MAE')
    #
    # axs[3].plot(model_epoch, model_mape, 'o')
    # axs[3].set_title('MAPE')
    # axs[3].set_xlabel('epoch')
    # axs[3].set_ylabel('MAPE')
    #
    # plt.savefig('./metrics_plot_test.png')
    # plt.show()


    model_r2 = np.zeros(len(model_paths))
    model_rmse = np.zeros(len(model_paths))
    model_mae = np.zeros(len(model_paths))
    model_mape = np.zeros(len(model_paths))
    model_epoch = np.zeros(len(model_paths))
    for idx, model_path in tqdm(enumerate(model_paths), nrows=40):
        epoch = re.findall(r'model_\d_(\d*).ckpt', model_path)[0]
        model, optimizer, epoch, accuracy_list = load_model(args.model, 0, lr,
                                                            data_dict['n_feats'],
                                                            data_dict['e_feats'],
                                                            emb_size=EMB_SIZE,
                                                            num_heads=NUM_HEADS)
        epoch = re.findall(r'model_\d_(\d*).ckpt', model_path)[0]
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.eval()
        all_preds = np.empty(len(data_dict['val']))
        all_labels = np.empty(len(data_dict['val']))
        for batch_id, batch_data in enumerate(data_dict['val']):
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

        test_r2 = r2_score(all_labels, all_preds)
        test_rmse = mean_squared_error(all_labels, all_preds, squared=False)
        test_mae = mean_absolute_error(all_labels, all_preds)
        test_mape = mean_absolute_percentage_error(all_labels, all_preds)

        model_r2[idx] = test_r2
        model_rmse[idx] = test_rmse
        model_mae[idx] = test_mae
        model_mape[idx] = test_mape
        model_epoch[idx] = int(epoch)

    fig, axs = plt.subplots(4, 1, constrained_layout=True)
    axs[0].plot(model_epoch, model_r2, 'o')
    axs[0].set_title('R2')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('R2')

    axs[1].plot(model_epoch, model_rmse, 'o')
    axs[1].set_title('RMSE')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('RMSE')

    axs[2].plot(model_epoch, model_mae, 'o')
    axs[2].set_title('MAE')
    axs[2].set_xlabel('epoch')
    axs[2].set_ylabel('MAE')

    axs[3].plot(model_epoch, model_mape, 'o')
    axs[3].set_title('MAPE')
    axs[3].set_xlabel('epoch')
    axs[3].set_ylabel('MAPE')

    plt.savefig('./metrics_plot_val.png')
    plt.show()
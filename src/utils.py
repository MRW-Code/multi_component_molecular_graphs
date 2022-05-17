import argparse
import os
import torch

parser = argparse.ArgumentParser(usage='python main.py')
parser.add_argument('--gpu_idx', action='store', dest='gpu_idx', default='0',
                  choices=['0', '1', '2', '3', '4', '5'])
parser.add_argument('--cpu', action='store_true',
                  dest='cpu', default=False)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx
device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
print(f'device = {device}')


import argparse
import os

parser = argparse.ArgumentParser(usage='python main.py')
parser.add_argument('--gpu_idx', action='store', dest='gpu_idx', default='0',
                  choices=['0', '1', '2', '3', '4', '5'])
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx


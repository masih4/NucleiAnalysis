import os

from utilsHover.utils import computeMetricsFromMasks
from Evaluate import main_mini
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="hovernext",
                    choices=["hovernext", "acs"], help='model')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--binaryThresh', type=float, default=0.5, help='random seed')
parser.add_argument('--hvThresh', default=[0.5, 0.5], help='random seed')
parser.add_argument('--dataset_path', type=str, default='1', help='path to dataset')
parser.add_argument('--pred_path', type=str, default='1', help='path to dataset')
parser.add_argument('--checkpoint_path', type=str, default='', help='path to model checkpoint')
parser.add_argument('--dataset_name', type=str, default='1', help='dataset name')
parser.add_argument('--results_dict', default={}, help='results dict')
args = parser.parse_args()

GT_pth = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/custom_split/NuInsSeg/train/'
# res = computeMetricsFromMasks(prediction_pth = prediction_pth, GT_pth = GT_pth)
model = 'hovernext'#'cellvit', ]

ths = [0.4, 0.5, 0.6, 0.7]
for fold in range(2,5):
    print('fold = ', fold)
    prediction_pth = f'/home/ntorbati/PycharmProjects/NucleiAnalysis/checkpoints1/nuinsseghovernext{fold}/fold{fold}Result'
    for th1 in ths:
        for th2 in ths:
            args.binaryThresh = th1
            args.hvThresh = [0.5,th2]
            args.model = model
            args.dataset_path = GT_pth
            args.checkpoint_path = f'/home/ntorbati/PycharmProjects/NucleiAnalysis/checkpoints1/nuinsseghovernext{fold}/checkpoint_epoch1.pth'
            args.pred_path = prediction_pth
            res = main_mini(args)
            print(th1, th2, res)
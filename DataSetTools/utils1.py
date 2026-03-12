import os
import numpy as np
import shutil
from sklearn.model_selection import KFold


def train_val_split(pth = None):
    if pth is None:
        pth = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/original_split/NuInsSeg/train/'


    data_pth = pth  # '/home/ntorbati/STORAGE/NucleiAnalysis/tif/1.kumar/train/'
    mask_pth = pth
    val_data_pth = os.path.join(pth, 'val')
    val_mask_pth = os.path.join(pth, 'val')


    os.makedirs(val_data_pth, exist_ok=True)
    os.makedirs(val_mask_pth, exist_ok=True)

    images = np.sort([im for im in os.listdir(data_pth) if im.endswith('.tif')])

    indices = np.arange(len(images))
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(kf.split(indices))
    val_index = indices[splits[0][1]]


    ## copy validation data for cellvit
    for im in images[val_index]:
        shutil.move(os.path.join(data_pth, im), os.path.join(val_data_pth, im))
        shutil.move(os.path.join(mask_pth, im.replace('.tif', '.npy')),
                    os.path.join(val_mask_pth, im.replace('.tif', '.npy')))

if __name__ == '__main__':
    pth = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/original_split/NuInsSeg/train/'
    train_val_split(pth=pth)
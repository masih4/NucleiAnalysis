import os
import numpy as np
import shutil
from sklearn.model_selection import KFold


def train_val_split(data_path = None, random_state = None, fold = None, conf_pth = None):

    config_pth = os.path.join(data_path, 'train')
    config_pth1 = os.path.join(data_path, 'train', 'fold0')
    data_pth = os.path.join(data_path, 'train', 'fold0', 'images')
    mask_pth = os.path.join(data_path, 'train', 'fold0', 'labels')
    val_data_pth = os.path.join(data_path, 'train', 'fold1', 'images')
    val_mask_pth = os.path.join(data_path, 'train', 'fold1', 'labels')

    try:
        shutil.rmtree(mask_pth)
        shutil.rmtree(val_data_pth)
        shutil.rmtree(val_mask_pth)
        shutil.rmtree(data_pth)
    except OSError as e:
        print(e)

    os.makedirs(data_pth, exist_ok=True)
    os.makedirs(mask_pth, exist_ok=True)

    os.makedirs(val_data_pth, exist_ok=True)
    os.makedirs(val_mask_pth, exist_ok=True)

    images = np.sort([im for im in os.listdir(data_path) if im.endswith('.tif')])

    indices = np.arange(len(images))
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(kf.split(indices))


    train_index = indices[splits[fold][0]]
    val_index = indices[splits[fold][1]]

    shutil.copy(os.path.join(conf_pth, 'weight_config.yaml'), os.path.join(config_pth, 'weight_config.yaml'))
    shutil.copy(os.path.join(conf_pth, 'dataset_config.yaml'), os.path.join(config_pth, 'dataset_config.yaml'))

    shutil.copy(os.path.join(conf_pth, 'weight_config.yaml'), os.path.join(config_pth1, 'weight_config.yaml'))
    shutil.copy(os.path.join(conf_pth, 'dataset_config.yaml'), os.path.join(config_pth1, 'dataset_config.yaml'))


    shutil.copy(os.path.join(conf_pth, 'cell_count.csv'), os.path.join(config_pth, 'cell_count.csv'))
    shutil.copy(os.path.join(conf_pth, 'types.csv'), os.path.join(config_pth, 'types.csv'))

    shutil.copy(os.path.join(conf_pth, 'cell_count.csv'), os.path.join(config_pth1, 'cell_count.csv'))
    shutil.copy(os.path.join(conf_pth, 'types.csv'), os.path.join(config_pth1, 'types.csv'))


    ## copy validation data for cellvit
    for im in images[train_index]:
        shutil.copy(os.path.join(data_path, im), os.path.join(data_pth, im))
        shutil.copy(os.path.join(data_path, im.replace('.tif', '.npy')),
                    os.path.join(mask_pth, im.replace('.tif', '.npy')))

    for im in images[val_index]:
        shutil.copy(os.path.join(data_path, im), os.path.join(val_data_pth, im))
        shutil.copy(os.path.join(data_path, im.replace('.tif', '.npy')),
                    os.path.join(val_mask_pth, im.replace('.tif', '.npy')))


if __name__ == '__main__':
    pth = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/custom_split/NuInsSeg/train/'
    train_val_split(data_path=pth,random_state=19,fold=0)
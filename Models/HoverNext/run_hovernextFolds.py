import numpy as np
from sklearn.model_selection import KFold

np.bool=np.bool_
from utils.train_model import train_model
from pathlib import Path
import os
import argparse
import random
import numpy as np
import torch
from Models.HoverNext.hover_next_train.src.multi_head_unet import get_model as get_hovernext
import tifffile
import matplotlib.pyplot as plt
from utils.utils_nuclei import gen_instance_hv_maps
def seed_torch(seed):
    if seed==None:
        seed= random.randint(1, 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="hovernext",
                    choices=["hovernext"], help='model')
parser.add_argument('--batch_size', type=int, default=5, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=512, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int, default=19, help='random seed')
parser.add_argument('--variant', type=str, default='1', help='fusion variant')
parser.add_argument('--iter', type=int, default=100, help='iteration')
parser.add_argument('--dataset_path', type=str, default='1', help='path to train dataset')
parser.add_argument('--validation_path', type=str, default='1', help='path to val dataset')
parser.add_argument('--modelSavePath', type=str, default='1', help='path to save best model')

parser.add_argument('--dataset_name', type=str, default='1', help='dataset name')
args = parser.parse_args()
seed_torch(args.seed)
def get_model(args):
    if args.model == "hovernext":
        model = get_hovernext(out_channels_cls=1,
                           out_channels_inst=5,
                           pretrained=True, )
    else:
        print('model error')
        return None
    return model

def main(args):
    model_name = args.model
    final_target_size = (args.img_size,args.img_size)
    n_class = args.num_classes
    device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## load data
    data_pth = args.dataset_path#os.path.join(args.dataset_path,'images') #'/home/ntorbati/STORAGE/NucleiAnalysis/tif/1.kumar/train/'
    mask_pth = args.dataset_path#os.path.join(args.dataset_path,'labels')
    # val_data_pth = args.validation_path#os.path.join(args.validation_path,'images')
    # val_mask_pth = args.validation_path#os.path.join(args.validation_path,'labels')
    # os.makedirs(val_data_pth,exist_ok=True)
    # os.makedirs(val_mask_pth,exist_ok=True)
    images = np.sort([im for im in os.listdir(data_pth) if im.endswith('.tif')])
    data = []
    masks = []
    for im in images:
        img = tifffile.imread(os.path.join(data_pth, im))
        data.append(img)
        msk = np.load(os.path.join(mask_pth, im.replace('.tif', '.npy')))
        masks.append(msk)
    data = np.stack(data)
    masks = np.stack(masks)
    ### add hv maps
    masks = gen_instance_hv_maps(masks)
    indices = np.arange(len(images))
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(kf.split(indices))

    for folds in range(0,n_folds):
        dir_checkpoint = Path(
            args.modelSavePath + args.dataset_name + args.model + str(
                folds) + '/')

        train_data = data[splits[folds][0]]
        train_masks = masks[splits[folds][0]]
        train_names = images[splits[folds][0]]

        val_data = data[splits[folds][1]]
        val_masks = masks[splits[folds][1]]
        val_names = images[splits[folds][1]]

        print(f'{args.model} training fold : {folds} ......................................................')
        iters = args.iter
        lr = 1e-4
        model1 = get_model(args)
        model1.to(device2)
        model1.n_classes = n_class
        target_size = final_target_size
        train_model(
        model = model1,
        device = device2,
        epochs = iters,
        batch_size = args.batch_size,
        learning_rate = lr,
        amp = False,
        weight_decay=0.7,  # learning rate decay rate
        target_siz=target_size,
        n_class=n_class,
        image_data1=train_data,
        mask_data1=train_masks,
        val_images = val_data,
        val_masks = val_masks,
        augmentation=True,# default None
        val_batch=1,
        early_stopping=100,
        ful_size=final_target_size,
        dir_checkpoint=dir_checkpoint,
        model_name=model_name,
        val_sleep_time = -1,
        nuclei=False,
            fold = folds,
            train_names = train_names,
            val_names = val_names,
            save_images=True
        )
        del model1
if __name__ == "__main__":
    args.model = "hovernext"
    dataset_tags = [
        # 'pcns',
        # 'monuseg',
        # 'cpm17',
        # 'tnbc',
        # 'cryonuseg',
        'nuinsseg',
        # 'consep',
        # 'puma',
        # 'monusac',
        # 'dsb'
    ]
    data_path = "/home/ntorbati/STORAGE/NucleiAnalysis/tif/custom_split/NuInsSeg/train/"
    logs_dir = '/home/ntorbati/PycharmProjects/NucleiAnalysis/Models/CellVit/CellViT/logs_paper/logs/'
    weight_path = '/home/ntorbati/PycharmProjects/NucleiAnalysis/checkpoints/'
    args.modelSavePath = weight_path
    dataset_name = dataset_tags[0]
    datasets_names = os.listdir(data_path)
    print(dataset_name)
    args.dataset_name = dataset_name
    args.dataset_path = data_path
    main(args)
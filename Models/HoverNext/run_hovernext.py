import numpy as np
np.bool=np.bool_
from utils.train_puma_dice import train_model
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
parser.add_argument('--img_size', type=int, default=256, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--variant', type=str, default='1', help='fusion variant')
parser.add_argument('--iter', type=int, default=300, help='iteration')
parser.add_argument('--dataset_path', type=str, default='1', help='path to train dataset')
parser.add_argument('--validation_path', type=str, default='1', help='path to val dataset')
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
    data_pth = os.path.join(args.dataset_path,'images') #'/home/ntorbati/STORAGE/NucleiAnalysis/tif/1.kumar/train/'
    mask_pth = os.path.join(args.dataset_path,'labels')
    val_data_pth = os.path.join(args.validation_path,'images')
    val_mask_pth = os.path.join(args.validation_path,'labels')
    os.makedirs(val_data_pth,exist_ok=True)
    os.makedirs(val_mask_pth,exist_ok=True)
    train_images = np.sort([im for im in os.listdir(data_pth) if im.endswith('.tif')])
    train_data = []
    train_masks = []
    for im in train_images:
        img = tifffile.imread(os.path.join(data_pth, im))
        train_data.append(img)
        msk = np.load(os.path.join(mask_pth, im.replace('.tif', '.npy')))
        train_masks.append(msk)
    train_data = np.stack(train_data)
    train_masks = np.stack(train_masks)
    val_images = np.sort([im for im in os.listdir(val_data_pth) if im.endswith('.tif')])
    val_data = []
    val_masks = []
    for im in val_images:
        img = tifffile.imread(os.path.join(val_data_pth, im))
        val_data.append(img)
        msk = np.load(os.path.join(val_mask_pth, im.replace('.tif', '.npy')))
        val_masks.append(msk)
    val_data = np.stack(val_data)
    val_masks = np.stack(val_masks)
    ### add hv maps
    train_masks = gen_instance_hv_maps(train_masks)
    val_masks = gen_instance_hv_maps(val_masks)
    for folds in [0]:#range(0,n_folds):
        print(f'{args.model} training fold : {folds} ......................................................')
        dir_checkpoint = Path('/home/ntorbati/PycharmProjects/NucleiAnalysis/checkpoints/' + args.dataset_name + model_name + str(folds) + '/')
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
        )
        del model1
if __name__ == "__main__":
    args.model = "hovernext"
    dataset_tags = ['pan-cancer-nuclei-seg', 'monuseg', 'cpm17', 'tnbc',
                    'cryonuseg', 'nuinsseg', 'consep', 'puma', 'monusac', 'data science bowl']
    data_path = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/'
    logs_dir = '/home/ntorbati/PycharmProjects/NucleiAnalysis/Models/CellVit/CellViT/logs_paper/logs'
    datasets_names = os.listdir(data_path)
    for dataset_tag in dataset_tags:
        for dataset_name in datasets_names:
            if dataset_tag in dataset_name:
                print(dataset_name)
                args.dataset_name = dataset_name
                args.dataset_path = os.path.join(data_path,dataset_name,'train','fold0')
                args.validation_path = os.path.join(data_path,dataset_name,'train','fold1')
                main(args)
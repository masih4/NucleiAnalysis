import numpy as np
np.bool=np.bool_
import os
import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="hovernext",
                    choices=["cellvit", "hovernext", "acs"], help='model')
parser.add_argument('--batch_size', type=int, default=5, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=256, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--variant', type=str, default='1', help='fusion variant')
parser.add_argument('--iter', type=int, default=300, help='iteration')
parser.add_argument('--dataset_path', type=str, default='1', help='path to dataset')
parser.add_argument('--dataset_name', type=str, default='1', help='dataset name')
parser.add_argument('--test_path', type=str, default='', help='test path')
parser.add_argument('--dataset_tag', type=str, default='1', help='dataset tag')
args = parser.parse_args()


def main(args):
    ## load data
    data_pth = args.dataset_path #'/home/ntorbati/STORAGE/NucleiAnalysis/tif/1.kumar/train/'
    images = glob.glob(f"{data_pth}/**/*.tif")
    masks = glob.glob(f"{data_pth}/**/*.npy")
    im_test = [os.path.join(args.test_path,im) for im in os.listdir(args.test_path) if (args.dataset_tag in im) and im.endswith(".tif")]
    msk_test = [os.path.join(args.test_path,im) for im in os.listdir(args.test_path) if (args.dataset_tag in im) and im.endswith(".npy")]
    images = images + im_test
    masks = masks + msk_test
    if len(images) != len(masks):
        print('wt...')
    # images = [im for im in os.listdir(data_pth) if im.endswith('.tif')]
    mask_data = []
    for im in images:
        msk = np.load(im.replace('.tif', '.npy'))
        mask_data.append(msk)

    number_images = len(images)
    num_inst = 0
    min_size = 1e10
    max_size = 0
    total_size = 0
    for msk in mask_data:
        if msk.shape[0] * msk.shape[1] < min_size:
            min_res = (msk.shape[0], msk.shape[1])
            min_size = msk.shape[0] * msk.shape[1]
        if msk.shape[0] * msk.shape[1] > max_size:
            max_res = (msk.shape[0], msk.shape[1])
            max_size = msk.shape[0] * msk.shape[1]
        num_inst += (len(np.unique(msk)) - 1)
        total_size += (msk.shape[0] * msk.shape[1])/(256*256)

    return [number_images, num_inst, min_res, max_res,num_inst/total_size]

def count_num_tiles(pth = ''):
    fld1 = os.path.join(pth, 'fold1','images')
    fld0 = os.path.join(pth, 'fold0','images')
    im0 = [im for im in os.listdir(fld0) if im.endswith('.tif')]
    im1 = [im for im in os.listdir(fld1) if im.endswith('.tif')]
    sum = len(im0) + len(im1)
    return sum



if __name__ == "__main__":
    dataset_tags = [
        # 'NuFuse-train',
        # 'NuFuse-test'#,
        'pan-cancer-nuclei-seg', 'monuseg', 'cpm17', 'tnbc',
                    'cryonuseg', 'nuinsseg', 'consep', 'puma', 'monusac', 'data science bowl'
                    ]
    data_path = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/'
    test_path = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/BigTestFolder'
    # logs_dir = '/home/ntorbati/PycharmProjects/NucleiAnalysis/Models/CellVit/CellViT/logs_paper/logs'
    datasets_names = os.listdir(data_path)
    dict = {}
    summ = []
    for dataset_tag in dataset_tags:
        for dataset_name in datasets_names:
            if dataset_tag in dataset_name:
                print(dataset_name)
                args.dataset_name = dataset_name
                args.dataset_path = os.path.join(data_path,dataset_name)
                args.test_path = test_path
                args.dataset_tag = dataset_tag

                stat = main(args)
                dict[dataset_tag] = stat
                print(dict[dataset_tag])
                args.dataset_path = os.path.join(data_path,dataset_name,'train')
                sum = count_num_tiles(args.dataset_path)
                summ.append(sum)
                print(dataset_name , '=', sum)
    summ = np.array(summ)
    print(np.sum(summ))
    print(dict)


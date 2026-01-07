import numpy as np
from pretrainedmodels.models.torchvision_models import model_name
np.bool=np.bool_
from utils.train_puma_dice import train_model
from sklearn.model_selection import KFold
from pathlib import Path
import os
import argparse
import random
import numpy as np
import torch
from Models.ACS.model import DualEncoderUNet
# from transformers import SegformerForSemanticSegmentation, SegformerConfig
# import segmentation_models_pytorch as smp
import yaml
from Models.HoverNext.hover_next_train.src.multi_head_unet import get_model as get_hovernext
from Models.get_models import get_train_model as get_cellvit
import tifffile
import matplotlib.pyplot as plt
from utils.utils_nuclei import gen_instance_hv_maps,get_fast_dice_2, get_fast_aji
# from utils.utils import split_patches
import cv2
import torch.nn.functional as F
from Models.CellVit.CellViT.cell_segmentation.utils.metrics import get_fast_pq, remap_label
from Models.CellVit.CellViT.cell_segmentation.utils.post_proc_cellvit import DetectionCellPostProcessor
from torchvision import transforms as T

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
                    choices=["hovernext", "acs"], help='model')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--dataset_path', type=str, default='1', help='path to dataset')
parser.add_argument('--checkpoint_path', type=str, default='', help='path to model checkpoint')
parser.add_argument('--dataset_name', type=str, default='1', help='dataset name')
parser.add_argument('--results_dict', default={}, help='results dict')
args = parser.parse_args()
seed_torch(args.seed)


def pad_to_multiple(x, multiple=32):
    """
    Pads the input tensor so that its height and width are divisible by 'multiple'.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).
        multiple (int): The value by which height and width should be divisible.

    Returns:
        padded (torch.Tensor): Padded tensor.
        pad (tuple): Amount of padding applied (pad_left, pad_right, pad_top, pad_bottom).
    """
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple  # extra rows needed
    pad_w = (multiple - w % multiple) % multiple  # extra columns needed

    # For simplicity, we pad only to the bottom and right
    pad_top, pad_left = 0, 0
    pad_bottom, pad_right = pad_h, pad_w

    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    return padded, (pad_left, pad_right, pad_top, pad_bottom)

def add_pad(x,pad_siz = (0,0)):
    padded = F.pad(x, (pad_siz[0], pad_siz[0], pad_siz[1], pad_siz[1]), mode='reflect')
    return padded, (pad_siz[0], pad_siz[0], pad_siz[1], pad_siz[1])


def remove_pad(x, pad):
    """
    Removes padding from the tensor.

    Args:
        x (torch.Tensor): Tensor after processing (B, C, H_padded, W_padded).
        pad (tuple): Padding amounts (pad_left, pad_right, pad_top, pad_bottom).

    Returns:
        cropped (torch.Tensor): Tensor cropped back to original size.
    """
    pad_left, pad_right, pad_top, pad_bottom = pad
    # If no padding was applied, simply return x
    if pad_bottom > 0:
        x = x[:, :, : -pad_bottom, :]
    if pad_right > 0:
        x = x[:, :, :, : -pad_right]
    if pad_top > 0:
        x = x[:, :, pad_top:, :]
    if pad_left > 0:
        x = x[:, :, :, pad_left:]
    return x


def np2torch(image, mask):
    image = image.astype("float32")
    mask = mask.astype("float32")
    image = image / 255
    image = np.transpose(image, (2, 0, 1))
    # temp = temp.astype("int32")

    # if np.sum(temp > self.n_class-1) or np.sum(temp < 0):
    #     temp[temp > self.n_class-1] = 0
    #     temp[temp < 0] = 0

    image = image.astype('float32')

    image = torch.tensor(image, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.float32)
    return image, mask


def get_model(args):

    if args.model == "acs":
        IgnoreBottleNeck = True
        segformer_variant = "nvidia/segformer-b2-finetuned-ade-512-512"
        model = DualEncoderUNet(
                                 segformer_variant=segformer_variant,
                                 cof_unet=1,
                                cof_seg=1,
                                 simple_fusion=1,
                                 regression=False,
                                instance_segmentation=True,
                                 classes=args.num_classes,
                                 in_channels=3,
                                 model_depth=4,
                                 unet_encoder_weights="imagenet",
                                 unet_encoder_name="convnext",
                                 IgnoreBottleNeck=IgnoreBottleNeck,
                                 decoder_channels=(256, 128, 64, 32, 16),
                                 )
        checkpoint_path = args.checkpoint_path
        cp = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(cp, strict=True)  # strict=True to ensure all trained layers match
    elif args.model == "hovernext":
        checkpoint_path = args.checkpoint_path
        model = get_hovernext(out_channels_cls=1,
                           out_channels_inst=5,
                           pretrained=True, )
        cp = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(cp, strict=True)  # strict=True to ensure all trained layers match

    elif args.model == "cellvit":
        config_path = "/home/ntorbati/PycharmProjects/NucleiAnalysis/Models/CellVit/CellViT/configs/examples/cell_segmentation/train_cellvit.yaml"
        with open(config_path, "r") as dataset_config_file:
            yaml_config = yaml.safe_load(dataset_config_file)
            dataset_config = dict(yaml_config)
        model = get_cellvit(dataset_config,pretrained_model=args.checkpoint_path)
    else:
        print('model error')
        return None

    return model


mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
inference_transforms_vit = T.Compose(
            [T.ToTensor(), T.Normalize(mean=mean, std=std)]
        )



mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
inference_transforms_cnn = T.Compose(
            [T.ToTensor(), T.Normalize(mean=mean, std=std)]
        )

def main(args):
    k = 0
    device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1 = get_model(args)
    model1.eval()
    model1.to(device2)


    ## load data
    data_pth = args.dataset_path
    mask_pth = args.dataset_path
    imgs_paths = np.sort([im for im in os.listdir(data_pth) if im.endswith('.tif')])
    image_data = []
    mask_data = []
    for im in np.sort(imgs_paths):
        img = tifffile.imread(os.path.join(data_pth, im))
        image_data.append(img)
        msk = np.load(os.path.join(mask_pth, im.replace('.tif', '.npy')))
        mask_data.append(msk)

    # mask_data = np.stack(mask_data)
    # image_data = np.stack(image_data)

    ### add hv maps




    ### tile and pad the images to 256 256
    # split_dev = 4
    # image_data = split_patches(image_data,split_dev)
    # mask_data = np.transpose(split_patches(np.transpose(mask_data, (0,2,3,1)),split_dev), (0,3,1,2))
    validation_pq = DetectionCellPostProcessor()
    SQ = []
    DQ = []
    PQ = []
    precision = []
    recall = []
    AJI = []
    DICE = []
    results_dict = args.results_dict
    with torch.no_grad():
        for images, true_masks, pth in zip(image_data, mask_data, imgs_paths):
            if 'cellseg' in pth:
                images = cv2.resize(images, (
                int(images.shape[1] * (40 / 200)), int(images.shape[0] * (40 / 200))))
                true_masks = cv2.resize(true_masks, (
                int(true_masks.shape[1] * (40 / 200)), int(true_masks.shape[0] * (40 / 200))),
                                  interpolation=cv2.INTER_NEAREST)
            true_masks = gen_instance_hv_maps(true_masks[np.newaxis, ...])


            if args.model == 'cellvit':
                images = inference_transforms_vit(images)
                true_masks = true_masks.astype("float32")
                true_masks = torch.tensor(true_masks, dtype=torch.float32)
            else:
                images = inference_transforms_cnn(images)
                true_masks = true_masks.astype("float32")
                true_masks = torch.tensor(true_masks, dtype=torch.float32)
                # images, true_masks =np2torch(images, true_masks)

            images = images.to(device=device2, dtype=torch.float32).unsqueeze(0)
            true_masks = true_masks.to(device=device2, dtype=torch.float32)
            images, pad = pad_to_multiple(images)
            masks_pred = model1(images)

            if args.model == "hovernext" or args.model == "acs":
                masks_pred = remove_pad(masks_pred, pad)
                binary_pred = masks_pred[:, 5]
                binary_pred = (F.sigmoid(binary_pred) > 0.5).float()
                preds = torch.cat([binary_pred.unsqueeze(1), masks_pred[:, 0:2]], dim=1)
                preds = np.transpose(preds.cpu().numpy(), (0, 2, 3, 1))
            elif args.model == "cellvit":
                binary_pred = torch.argmax(masks_pred['nuclei_binary_map'], dim=1)
                # preds = masks_pred['hv_map'].permute(0,2,3,1).cpu().numpy()
                preds = torch.cat([binary_pred.unsqueeze(1), masks_pred['hv_map']], dim=1)
                preds = remove_pad(preds, pad)
                preds = np.transpose(preds.cpu().numpy(),(0,2,3,1))

            pred_inst, _ = validation_pq.post_process_cell_segmentation(pred_map=preds)
            pred_inst = remap_label(pred_inst)
            gt = remap_label(true_masks[0, 0].cpu().numpy().astype(np.int32))
            pq, pred_details = get_fast_pq(gt, pred_inst)
            if pq[0] != 0:
                dice = get_fast_dice_2(gt, pred_inst)
                aji = get_fast_aji(gt, pred_inst)
            else:
                dice = 0
                aji = 0

            tp = len(pred_details[0])
            fp = len(pred_details[3])
            fn = len(pred_details[2])
            DQ.append(pq[0])
            SQ.append(pq[1])
            PQ.append(pq[2])
            dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1.0e-6)  # good practice?

            precision.append(tp / (tp + fp + 1e-6))
            recall.append(tp / (tp + fn + 1e-6))
            AJI.append(aji)
            DICE.append(dice)
            for dataset_tag in dataset_tags:
                if dataset_tag in pth:
                    results_dict[dataset_tag]['PQ'].append(pq[2])
                    results_dict[dataset_tag]['DQ'].append(pq[0])
                    results_dict[dataset_tag]['SQ'].append(pq[1])
                    results_dict[dataset_tag]['AJI'].append(aji)
                    results_dict[dataset_tag]['DICE'].append(dice)
                    results_dict[dataset_tag]['precision'].append(tp / (tp + fp + 1e-6))
                    results_dict[dataset_tag]['recall'].append(tp / (tp + fn + 1e-6))

        for dataset_tag in dataset_tags:

            results_dict[dataset_tag]['PQ'] = np.mean(np.stack(results_dict[dataset_tag]['PQ']))
            results_dict[dataset_tag]['DQ'] = np.mean(np.stack(results_dict[dataset_tag]['DQ']))
            results_dict[dataset_tag]['SQ'] = np.mean(np.stack(results_dict[dataset_tag]['SQ']))
            results_dict[dataset_tag]['precision'] = np.mean(np.stack(results_dict[dataset_tag]['precision']))
            results_dict[dataset_tag]['recall'] = np.mean(np.stack(results_dict[dataset_tag]['recall']))
            results_dict[dataset_tag]['DICE'] = np.mean(np.stack(results_dict[dataset_tag]['DICE']))
            results_dict[dataset_tag]['AJI'] = np.mean(np.stack(results_dict[dataset_tag]['AJI']))

        SQ = np.mean(np.stack(SQ))  # Divide by total validation samples
        DQ = np.mean(np.stack(DQ))  # Divide by total validation samples
        PQ = np.mean(np.stack(PQ))  # Divide by total validation samples
        precision = np.mean(np.stack(precision))  # Divide by total validation samples
        recall = np.mean(np.stack(recall))  # Divide by total validation samples
        aji = np.mean(np.stack(AJI))
        dice = np.mean(np.stack(DICE))
        results_dict['overall']['PQ'] = PQ
        results_dict['overall']['DQ'] = DQ
        results_dict['overall']['SQ'] = SQ
        results_dict['overall']['precision'] = precision
        results_dict['overall']['recall'] = recall
        results_dict['overall']['AJI'] = aji
        results_dict['overall']['DICE'] = dice
        print(args.dataset_name,
            f"PQ: {PQ:.4f}",f"AJI: {aji:.4f}",f"Dice: {dice:.4f}", f"SQ: {SQ:.4f}", f"DQ: {DQ:.4f}", f"precision: {precision:.4f}", f"recall: {recall:.4f}")


        for dataset_tag in dataset_tags:
            print(dataset_tag, results_dict[dataset_tag])
    return results_dict


dataset_tags = ['pan-cancer-nuclei-seg', 'puma', 'monuseg', 'cpm17', 'monusac',
                    'consep', 'tnbc', 'cryonuseg', 'data science bowl', 'nuinsseg']
metrics = ['precision', 'recall', 'SQ', 'DQ', 'PQ','AJI','DICE']


if __name__ == "__main__":
    models = ['cellvit','hovernext']
    for model in models:
        args.model = model
        model_name = args.model
        overal = {}
        overal[model_name] = {}
        overal[model_name]['PQ'] = []
        overal[model_name]['DQ'] = []
        overal[model_name]['SQ'] = []
        overal[model_name]['precision'] = []
        overal[model_name]['recall'] = []
        overal[model_name]['AJI'] = []
        overal[model_name]['DICE'] = []
        data_path = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/'
        logs_dir = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/BigTestFolder/'

        datasets_names = os.listdir(data_path)

        results_dict_all = []
        for dataset_tag in dataset_tags:
            for dataset_name in datasets_names:
                if dataset_tag in dataset_name:
                    print(dataset_name)
                    results_dict = {}
                    results_dict['overall'] = {}
                    for metric in metrics:
                        results_dict['overall'][metric] = []

                    for dataset_tag1 in dataset_tags:
                        results_dict[dataset_tag1] = {}
                        for metric in metrics:
                            results_dict[dataset_tag1][metric] = []
                    args.results_dict = results_dict

                    args.dataset_name = dataset_name
                    args.dataset_path = os.path.join(data_path,'BigTestFolder')
                    if args.model == 'hovernext' or args.model == 'acs':
                        checkpoint_path = os.path.join('/home/ntorbati/PycharmProjects/NucleiAnalysis/checkpoints/', args.model)
                        args.checkpoint_path = os.path.join(checkpoint_path,f'{dataset_name}', 'checkpoint_epoch1.pth')
                    elif args.model == 'cellvit':
                        checkpoint_path = os.path.join('/home/ntorbati/PycharmProjects/NucleiAnalysis/Models/CellVit/CellViT/logs_paper/logs/', dataset_name)
                        checkpoint_path = os.path.join(checkpoint_path, 'train')
                        sub_fld = [flds for flds in os.listdir(checkpoint_path) if '_None' in flds]

                        checkpoint_path = os.path.join(checkpoint_path, sub_fld[0] ,'checkpoints', 'model_best.pth')
                        args.checkpoint_path = checkpoint_path
                    results_dict = main(args)
                    overal[model_name]['PQ'].append(results_dict['overall']['PQ'])
                    overal[model_name]['DQ'].append(results_dict['overall']['DQ'])
                    overal[model_name]['SQ'].append(results_dict['overall']['SQ'])
                    overal[model_name]['AJI'].append(results_dict['overall']['AJI'])
                    overal[model_name]['DICE'].append(results_dict['overall']['DICE'])
                    overal[model_name]['precision'].append(results_dict['overall']['precision'])
                    overal[model_name]['recall'].append(results_dict['overall']['recall'])
                    results_dict_all.append(results_dict)
        overal[model_name]['PQ'] = (np.mean(overal[model_name]['PQ']))
        overal[model_name]['DQ'] = (np.mean(overal[model_name]['DQ']))
        overal[model_name]['SQ'] = (np.mean(overal[model_name]['SQ']))
        overal[model_name]['AJI'] = (np.mean(overal[model_name]['AJI']))
        overal[model_name]['DICE'] = (np.mean(overal[model_name]['DICE']))
        overal[model_name]['precision'] = (np.mean(overal[model_name]['precision']))
        overal[model_name]['recall'] = (np.mean(overal[model_name]['recall']))
        np.save('/home/ntorbati/PycharmProjects/NucleiAnalysis/Test_Results/' + model_name + '.npy', results_dict_all)
        np.save('/home/ntorbati/PycharmProjects/NucleiAnalysis/Test_Results/' + model_name + 'overall.npy', overal)
        print(overal)


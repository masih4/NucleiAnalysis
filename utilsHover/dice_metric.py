import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import random
import torch.nn.functional as F
def tile_or_pad1(image, mask, target_size =(1024,1024),pad_val = 1,only_pad = False) :
    C, H, W = image.shape
    target_h, target_w = target_size

    # Case 1: Exact match
    if H == target_h and W == target_w:
        return [(image, mask)]

    # Case 2: Both smaller — center pad
    elif H <= target_h and W <= target_w:
        pad_h = target_h - H
        pad_w = target_w - W

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=pad_val)
        mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return [(image, mask)]

    # Case 3: One dimension smaller — center pad it, crop the other
    else:
        if only_pad:
            return [(image, mask)]
        else:
            if H < target_h:
                pad_h = target_h - H
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                image = F.pad(image, (0, 0, pad_top, pad_bottom), mode='constant', value=pad_val)
                mask = F.pad(mask, (0, 0, pad_top, pad_bottom), mode='constant', value=0)
                H = target_h

            if W < target_w:
                pad_w = target_w - W
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                image = F.pad(image, (pad_left, pad_right, 0, 0), mode='constant', value=pad_val)
                mask = F.pad(mask, (pad_left, pad_right, 0, 0), mode='constant', value=0)
                W = target_w

            # Crop randomly from larger side(s)
            y = random.randint(0, H - target_h) if H > target_h else 0
            x = random.randint(0, W - target_w) if W > target_w else 0

            image = image[:, y:y + target_h, x:x + target_w]
            mask = mask[y:y + target_h, x:x + target_w]

            return [(image, mask)]
def calculate_dice_from_masks(mask1, mask2, eps=0.00001):
    """Calculate the DICE score between two binary masks."""
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1) + np.sum(mask2)
    dice_score = (2 * intersection + eps) / (union + eps)
    return dice_score

def calculate_dice_score_with_masks(tif1, tif2, image_shape, eps=0.00001,nuclei = False):
    """Calculate the DICE score between two TIF files using masks."""
    tif1 = np.array(Image.open(tif1).resize(image_shape, Image.NEAREST))
    tif2 = np.array(Image.open(tif2).resize(image_shape, Image.NEAREST))

    if tif1.shape != tif2.shape:
        im1 = tile_or_pad1(torch.tensor(np.zeros((tif1.shape[0], tif1.shape[1], 3))).permute(2, 0, 1),
                          torch.tensor(tif1), target_size=(1024, 1024),
                          pad_val=0, only_pad=True)
        tif1 = im1[0][1].cpu().numpy().astype(np.uint8)
    # If the ground truth (tif1) has 4 channels, use the first channel
    if tif1.ndim == 3 and tif1.shape[-1] == 4:  # Check if it's 4 channels
        tif1 = tif1[:, :, 0]  # Use the first channel (or modify as needed for your use case)

    # If the predictions (tif2) have multiple channels, use the first channel
    if tif2.ndim == 3 and tif2.shape[-1] > 1:
        tif2 = tif2[:, :, 0]  # Use the first channel (or modify as needed)




    dice_scores = {}
    if nuclei == 10:
        class_map = {
            1:"nuclei_endothelium",
            2:"nuclei_plasma_cell",
            3:"nuclei_stroma",
            4:"nuclei_tumor",
            5:"nuclei_histiocyte",
            6:"nuclei_apoptosis",
            7:"nuclei_epithelium",
            8:"nuclei_melanophage",
            9:"nuclei_neutrophil",
            10:"nuclei_lymphocyte",
        }
    elif nuclei == 3:
        class_map = {
            1:"cell_lymphocyte",
            2:"cell_tumor",
            3:"cell_other",
        }
    elif nuclei == 2:
        class_map = {
            1:"cell_other",
            2:"cell_tumor",
        }
    elif nuclei == 5:
        class_map = {1:'Epithelial',
                       2:'Lymphocyte',
                       3:'Macrophage',
                     4: 'Neutrophil',
                     5: 'Ambiguous'
                     }
    else:
        class_map = {1: 'tissue_stroma', 2: 'tissue_blood_vessel', 3: 'tissue_tumor', 4: 'tissue_epidermis', 5: 'tissue_necrosis'}

    for category in range(1, len(class_map) + 1):
        # Generate binary masks for each class
        mask1 = np.where(tif1 == category, 1, 0)
        mask2 = np.where(tif2 == category, 1, 0)

        # If both masks are empty, perfect match
        if np.sum(mask1) == 0 and np.sum(mask2) == 0:
            dice_score = 1.0
        else:
            dice_score = calculate_dice_from_masks(mask1, mask2, eps)

        dice_scores[class_map[category]] = dice_score

    return dice_scores

def calculate_dice_for_files(ground_truth_file, prediction_file, image_shape,nuclei=False):
    """Calculate the DICE scores for a single ground truth and prediction file."""
    dice_scores = calculate_dice_score_with_masks(ground_truth_file, prediction_file, image_shape,nuclei = nuclei)

    # Calculate the average DICE score across all classes for this file
    class_scores = [score for score in dice_scores.values() if score is not None]
    average_dice = sum(class_scores) / len(class_scores) if class_scores else 0.0
    dice_scores['average_dice'] = average_dice

    return dice_scores

def compute_dice_scores(gt_folder, pred_folder, image_shape=(1024, 1024),nuclei = False):
    """Read ground truth and prediction TIF images, compute Dice scores for all pairs."""
    # Get sorted lists of ground truth and prediction files
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(".tif")])
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(".tif")])

    # Ensure both folders have the same number of files
    # if len(gt_files) != len(pred_files):
    #     raise ValueError("Mismatch in the number of files between ground truth and predictions.")
    mean_dice = 0
    counter=0
    if nuclei:
        mean_dice_classes = np.zeros(nuclei)
    else:
        mean_dice_classes = np.zeros(5)
    # Compute Dice scores for each file pair
    overall_scores = {}
    if nuclei == 10:
        class_map = {
            1:"nuclei_endothelium",
            2:"nuclei_plasma_cell",
            3:"nuclei_stroma",
            4:"nuclei_tumor",
            5:"nuclei_histiocyte",
            6:"nuclei_apoptosis",
            7:"nuclei_epithelium",
            8:"nuclei_melanophage",
            9:"nuclei_neutrophil",
            10:"nuclei_lymphocyte",
        }
    elif nuclei == 3:
        class_map = {
            1:"cell_lymphocyte",
            2:"cell_tumor",
            3:"cell_other",
        }
    elif nuclei == 2:
        class_map = {
            1:"cell_other",
            2:"cell_tumor",
        }
    elif nuclei == 5:
        class_map = {1:'Epithelial',
                       2:'Lymphocyte',
                       3:'Macrophage',
                       4:'Neutrophil',
                     5:'Ambiguous'
                       }

    else:
        class_map = {1: 'tissue_stroma', 2: 'tissue_blood_vessel', 3: 'tissue_tumor', 4: 'tissue_epidermis', 5: 'tissue_necrosis'}

    for gt_file in gt_files:
        gt_path = os.path.join(gt_folder, gt_file)
        pred_path = os.path.join(pred_folder, gt_file.replace('_tissue.tif', '.tif'))

        # print(f"Processing: GT={gt_file}, Pred={pred_file}")

        # Calculate Dice scores for the pair
        scores = calculate_dice_for_files(gt_path, pred_path, image_shape,nuclei)
        overall_scores[gt_file] = scores
        mean_dice += overall_scores[gt_file]['average_dice']
        for ds in range(len(class_map)):
            mean_dice_classes[ds] += scores[class_map[ds+1]]


        counter += 1

        # print(f"Dice Scores for {gt_file}: {scores}")

    return overall_scores, mean_dice/counter,mean_dice_classes/counter


def calculate_micro_dice_score_with_masks(gt_folder, pred_folder, image_shape, eps=0.00001, nuclei = False):
    """
    Calculate the overall micro DICE score across all classes between two folders of TIF masks.

    Args:
        gt_folder (str): Path to the folder containing ground truth TIF masks.
        pred_folder (str): Path to the folder containing predicted TIF masks.
        image_shape (tuple): Shape to resize the images (height, width).
        eps (float): Small value to avoid division by zero.

    Returns:
        dict: Micro DICE scores for each class and the average micro DICE score.
    """
    if nuclei == 10:
        class_map = {
            1:"nuclei_endothelium",
            2:"nuclei_plasma_cell",
            3:"nuclei_stroma",
            4:"nuclei_tumor",
            5:"nuclei_histiocyte",
            6:"nuclei_apoptosis",
            7:"nuclei_epithelium",
            8:"nuclei_melanophage",
            9:"nuclei_neutrophil",
            10:"nuclei_lymphocyte",
        }
    elif nuclei == 3:
        class_map = {
            1:"cell_lymphocyte",
            2:"cell_tumor",
            3:"cell_other",
        }

    else:
        class_map = {1: 'tissue_stroma', 2: 'tissue_blood_vessel', 3: 'tissue_tumor', 4: 'tissue_epidermis', 5: 'tissue_necrosis'}

    total_gt_mask = {class_name: [] for class_name in class_map.values()}  # Ground truth masks
    total_pred_mask = {class_name: [] for class_name in class_map.values()}  # Predicted masks

    # Get sorted lists of files
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.tif')])
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith('.tif')])

    # Ensure files in both folders match
    # if len(gt_files) != len(pred_files):
    #     raise ValueError("Ground truth and prediction folders must contain the same number of files.")

    for gt_file in gt_files:
        gt_path = os.path.join(gt_folder, gt_file)
        pred_path = os.path.join(pred_folder, gt_file.replace('_tissue.tif', '.tif'))
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"Missing file: {gt_path} or {pred_path}")
            continue

        # Load and preprocess the TIF images
        gt_tif = np.array(Image.open(gt_path))#.resize(image_shape, Image.NEAREST))
        pred_tif = np.array(Image.open(pred_path).resize(image_shape, Image.NEAREST))
        # pred_tif = pred_tif


        if pred_tif.shape != gt_tif.shape:
            im1 = tile_or_pad1(torch.tensor(np.zeros((gt_tif.shape[0], gt_tif.shape[1], 3))).permute(2, 0, 1),
                               torch.tensor(gt_tif), target_size=(1024, 1024),
                               pad_val=0, only_pad=True)
            gt_tif = im1[0][1].cpu().numpy().astype(np.uint8)


        # If ground truth or prediction has more than one channel, use the first channel
        if gt_tif.ndim == 3 and gt_tif.shape[-1] > 1:
            gt_tif = gt_tif[:, :, 0]
        if pred_tif.ndim == 3 and pred_tif.shape[-1] > 1:
            pred_tif = pred_tif[:, :, 0]

        # Accumulate masks for each class
        for category, class_name in class_map.items():
            gt_mask = np.where(gt_tif == category, 1, 0)
            pred_mask = np.where(pred_tif == category, 1, 0)

            total_gt_mask[class_name].append(gt_mask)
            total_pred_mask[class_name].append(pred_mask)

    # Concatenate all masks for each class
    for class_name in class_map.values():
        if total_gt_mask[class_name] and total_pred_mask[class_name]:  # Avoid empty lists
            # total_gt_mask[class_name] = np.concatenate(total_gt_mask[class_name], axis=0)
            # total_pred_mask[class_name] = np.concatenate(total_pred_mask[class_name], axis=0)
            a = 0
        else:
            total_gt_mask[class_name] = [np.zeros(1)]
            total_pred_mask[class_name] = [np.zeros(1)]

    # Calculate the micro DICE score for each class
    micro_dice_scores = {}
    for class_name in class_map.values():
        intersection = 0
        union = 0
        for ind in range(len(total_gt_mask[class_name])):
            mask1 = total_gt_mask[class_name][ind]
            mask2 = total_pred_mask[class_name][ind]

            intersection += np.sum(mask1 & mask2)
            union += np.sum(mask1) + np.sum(mask2)

        dice_score = (2 * intersection + eps) / (union + eps)
        if intersection == 0:
            dice_score = 0.0
        if union == 0:
            dice_score = 1.0

        micro_dice_scores[class_name] = dice_score

    # Calculate the average micro DICE score across all classes
    average_dice_score = np.mean(list(micro_dice_scores.values()))
    micro_dice_scores['average_micro_dice'] = average_dice_score

    return average_dice_score, list(micro_dice_scores.values())


def calculate_micro_dice_score_with_masks_eval(gt_folder, pred_folder, image_shape, eps=0.00001, nuclei = False):
    """
    Calculate the overall micro DICE score across all classes between two folders of TIF masks.

    Args:
        gt_folder (str): Path to the folder containing ground truth TIF masks.
        pred_folder (str): Path to the folder containing predicted TIF masks.
        image_shape (tuple): Shape to resize the images (height, width).
        eps (float): Small value to avoid division by zero.

    Returns:
        dict: Micro DICE scores for each class and the average micro DICE score.
    """
    if nuclei == 10:
        class_map = {
            1:"nuclei_endothelium",
            2:"nuclei_plasma_cell",
            3:"nuclei_stroma",
            4:"nuclei_tumor",
            5:"nuclei_histiocyte",
            6:"nuclei_apoptosis",
            7:"nuclei_epithelium",
            8:"nuclei_melanophage",
            9:"nuclei_neutrophil",
            10:"nuclei_lymphocyte",
        }
    elif nuclei == 3:
        class_map = {
            1:"cell_lymphocyte",
            2:"cell_tumor",
            3:"cell_other",
        }

    else:
        class_map = {1: 'tissue_stroma', 2: 'tissue_blood_vessel', 3: 'tissue_tumor', 4: 'tissue_epidermis'}#, 5: 'tissue_necrosis'}

    total_gt_mask = {class_name: [] for class_name in class_map.values()}  # Ground truth masks
    total_pred_mask = {class_name: [] for class_name in class_map.values()}  # Predicted masks

    # Get sorted lists of files
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.tif')])
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith('.tif')])

    # Ensure files in both folders match
    # if len(gt_files) != len(pred_files):
    #     raise ValueError("Ground truth and prediction folders must contain the same number of files.")

    for gt_file in gt_files:
        gt_path = os.path.join(gt_folder, gt_file)
        pred_path = os.path.join(pred_folder, gt_file.replace('_tissue.tif', '.tif'))
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"Missing file: {gt_path} or {pred_path}")
            continue

        # Load and preprocess the TIF images
        gt_tif = np.array(Image.open(gt_path))#.resize(image_shape, Image.NEAREST))
        pred_tif = np.array(Image.open(pred_path).resize(image_shape, Image.NEAREST))
        # pred_tif = pred_tif


        if pred_tif.shape != gt_tif.shape:
            im1 = tile_or_pad1(torch.tensor(np.zeros((gt_tif.shape[0], gt_tif.shape[1], 3))).permute(2, 0, 1),
                               torch.tensor(gt_tif), target_size=(1024, 1024),
                               pad_val=0, only_pad=True)
            gt_tif = im1[0][1].cpu().numpy().astype(np.uint8)


        # If ground truth or prediction has more than one channel, use the first channel
        if gt_tif.ndim == 3 and gt_tif.shape[-1] > 1:
            gt_tif = gt_tif[:, :, 0]
        if pred_tif.ndim == 3 and pred_tif.shape[-1] > 1:
            pred_tif = pred_tif[:, :, 0]

        # Accumulate masks for each class
        for category, class_name in class_map.items():
            gt_mask = np.where(gt_tif == category, 1, 0)
            pred_mask = np.where(pred_tif == category, 1, 0)

            total_gt_mask[class_name].append(gt_mask)
            total_pred_mask[class_name].append(pred_mask)

    # Concatenate all masks for each class
    for class_name in class_map.values():
        if total_gt_mask[class_name] and total_pred_mask[class_name]:  # Avoid empty lists
            # total_gt_mask[class_name] = np.concatenate(total_gt_mask[class_name], axis=0)
            # total_pred_mask[class_name] = np.concatenate(total_pred_mask[class_name], axis=0)
            a = 0
        else:
            total_gt_mask[class_name] = [np.zeros(1)]
            total_pred_mask[class_name] = [np.zeros(1)]

    metrics = {}

    for class_name in class_map.values():
        tp = 0
        fp = 0
        fn = 0

        for ind in range(len(total_gt_mask[class_name])):
            mask_gt = total_gt_mask[class_name][ind].astype(bool)
            mask_pred = total_pred_mask[class_name][ind].astype(bool)

            tp += np.sum(mask_gt & mask_pred)   # True Positives
            fp += np.sum(~mask_gt & mask_pred)  # False Positives
            fn += np.sum(mask_gt & ~mask_pred)  # False Negatives

        # Dice
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        if tp == 0 and fp == 0 and fn == 0:
            dice = 1.0  # empty masks, treat as perfect

        # IoU (Jaccard)
        iou = (tp + eps) / (tp + fp + fn + eps)

        metrics[class_name] = {
            "dice": dice,
            "iou": iou,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    # Compute averages across all classes
    avg_dice = np.mean([m["dice"] for m in metrics.values()])
    avg_iou = np.mean([m["iou"] for m in metrics.values()])

    metrics["average"] = {
        "dice": avg_dice,
        "iou": avg_iou,
        "tp": sum(m["tp"] for m in metrics.values()),
        "fp": sum(m["fp"] for m in metrics.values()),
        "fn": sum(m["fn"] for m in metrics.values()),
    }

    return metrics
    # Calculate the micro DICE score for each class
    # micro_dice_scores = {}
    # for class_name in class_map.values():
    #     intersection = 0
    #     union = 0
    #     for ind in range(len(total_gt_mask[class_name])):
    #         mask1 = total_gt_mask[class_name][ind]
    #         mask2 = total_pred_mask[class_name][ind]
    #
    #         intersection += np.sum(mask1 & mask2)
    #         union += np.sum(mask1) + np.sum(mask2)
    #
    #     dice_score = (2 * intersection + eps) / (union + eps)
    #     if intersection == 0:
    #         dice_score = 0.0
    #     if union == 0:
    #         dice_score = 1.0
    #
    #     micro_dice_scores[class_name] = dice_score
    #
    # # Calculate the average micro DICE score across all classes
    # average_dice_score = np.mean(list(micro_dice_scores.values()))
    # micro_dice_scores['average_micro_dice'] = average_dice_score
    #
    # return average_dice_score, list(micro_dice_scores.values())


# Example Usage
if __name__ == "__main__":
    ground_truth_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_images"
    prediction_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_prediction"
    image_shape = (1024, 1024)  # Adjust based on your images' resolution

    dice_scores = compute_dice_scores(ground_truth_folder, prediction_folder, image_shape)
    # print("Overall Dice Scores:")
    # for file, scores in dice_scores.items():
    #     print(f"{file}: {scores}")

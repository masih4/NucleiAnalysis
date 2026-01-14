from typing import List
import segmentation_models_pytorch as smp
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from numpy.core.defchararray import endswith
import csv
# from ocelot_eval import ocelot_f1_main
from utils.dice_puma import compute_dice_scores, calculate_micro_dice_score_with_masks_eval, calculate_micro_dice_score_with_masks
import os
import tifffile
import torch
from PIL import Image

import kornia.augmentation as K
# import kornia.geometry.transform as T
# import kornia.morphology as KM
import torchvision.transforms as T
import scipy
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
import shutil
from typing import List
import json
from shapely.geometry import shape
from rasterio.features import rasterize
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree
import xml.etree.ElementTree as ET
from scipy.ndimage import label
from collections import Counter
from scipy.optimize import linear_sum_assignment




from scipy.ndimage import measurements







# def Mine_resize(image = None, mask = None, final_size = None):
#     """Resize image and mask to the final size."""
#     image_resized = F.interpolate(image, size=(final_size[0], final_size[1]), mode="bilinear",
#                                   align_corners=False)
#     mask_resized = F.interpolate(mask.unsqueeze(1).float(), size=final_size, mode="nearest").squeeze(1).long()
#     return image_resized, mask_resized
#





class RandomOneOf(torch.nn.Module):
    def __init__(self, augmentations, p=1.0):
        super().__init__()
        self.augmentations = torch.nn.ModuleList(augmentations)
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() > self.p:
            return x  # No augmentation applied
        # Randomly select one augmentation
        idx = torch.randint(0, len(self.augmentations), (1,)).item()
        return self.augmentations[idx](x)


# class KorniaAugmentation:
#     def __init__(self, mode="train", num_classes=6, seed=None, size = None):
#         self.mode = mode
#         self.size = size
#         self.num_classes = num_classes
#         self.seed = seed
#         torch.manual_seed(seed) if seed else None
#
#         # Define PiecewiseAffine augmentation
#         self.piecewise_affine = K.RandomThinPlateSpline(scale=0.15, align_corners=False)
#
#         # Define Shape Transformations
#         # self.affine = T.Compose([
#         #     T.RandomAffine(
#         #         degrees=(-179, 179),  # Rotation between -179 and 179 degrees
#         #         translate=(0.01, 0.01),  # Translation up to 1% of image dimensions
#         #         scale=(0.8, 1.2),  # Scaling between 0.8 and 1.2
#         #         shear=(-5, 5),  # Shear range between -5 and 5 degrees
#         #         # interpolation=InterpolationMode.NEAREST,  # Use nearest neighbor (0 corresponds to nearest)
#         #         fill=0  # Fill padding areas with black (0)
#         #     )
#         # ])
#
#         # Create a RandomOneOf augmentation for the shape transformations
#         self.shape_augs = torch.nn.Sequential(#RandomOneOf([
#             # K.RandomThinPlateSpline(p=0.5, scale=0.1, same_on_batch=True, keepdim=True),  # Correct parameters
#             T.RandomAffine(
#                 degrees=(-179, 179),  # Rotation between -179 and 179 degrees
#                 translate=(0.05, 0.05),  # Translation up to 1% of image dimensions
#                 scale=(0.8, 1.2),  # Scaling between 0.8 and 1.2
#                 shear=(-5, 5),  # Shear range between -5 and 5 degrees
#                 # interpolation=InterpolationMode.NEAREST,  # Use nearest neighbor (0 corresponds to nearest)
#                 fill=0  # Fill padding areas with black (0)
#             ),
#             K.RandomHorizontalFlip(p=0.5),
#             K.RandomVerticalFlip(p=0.5),
#             # K.CenterCrop(size=size)
#         )
#
#         # Define Input Transformations
#         self.input_augs = torch.nn.Sequential(
#             RandomOneOf([
#                 K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
#                 K.RandomMedianBlur((3, 3), p=1.0),
#                 K.RandomGaussianNoise(mean=0.0, std=0.05, p=1.0),
#             ], p=1.0),
#             K.RandomBrightness(brightness=(0.9, 1.1), p=1.0),
#             K.RandomContrast(contrast=(0.75, 1.25), p=1.0),
#             K.RandomHue(hue=(-0.05, 0.05), p=1.0),
#             K.RandomSaturation(saturation=(0.8, 1.2), p=1.0),
#         )
#
#     def __apply_piecewise_affine(self, images, masks):
#         """Apply piecewise affine transformation to both images and masks."""
#         # Stack masks into additional channels
#
#         B, C, H, W = images.shape
#         masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
#
#         # Create a temporary ones tensor to track padding
#         ones_tensor = torch.ones_like(images)
#
#         # Combine images, masks, and the ones tensor for padding tracking
#         combined = torch.cat([images, masks_one_hot, ones_tensor], dim=1)
#
#         # Apply affine transformation to the combined tensor (image + mask + ones)
#         combined_augmented = self.piecewise_affine(combined)
#
#         # Separate images, masks, and ones tensor after transformation
#         images_aug = combined_augmented[:, :C]
#         masks_aug = combined_augmented[:, C:C + self.num_classes]
#         ones_aug = combined_augmented[:, C + self.num_classes:]
#
#
#         masks_aug = masks_aug.argmax(dim=1).long()  # Convert back to class labels
#
#         # Track padding by checking where ones_aug has been turned to 0 (padding areas)
#         padding_mask = ones_aug == 0  # This will be True in the padding areas
#
#         # Fill padding for images with 1 (where padding_mask is True)
#         images_aug = torch.where(padding_mask, torch.tensor(1.0, device=images_aug.device), images_aug)
#         return images_aug, masks_aug
#
#     def __apply_erode_margins(self, masks):
#         """Add margins to masks by applying erosion."""
#         B, H, W = masks.shape
#         masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
#
#         # Apply erosion to each class
#         kernel = torch.ones((3, 3), device=masks.device)
#         eroded_channels = []
#         for i in range(self.num_classes):
#             mask_channel = masks_one_hot[:, i:i + 1]
#             eroded = mask_channel
#             for _ in range(5):  # Perform erosion 3 times
#                 eroded = KM.erosion(eroded, kernel)
#             eroded_channels.append(eroded)
#
#         # Combine eroded masks
#         eroded_masks = torch.cat(eroded_channels, dim=1)
#         combined_masks = eroded_masks.argmax(dim=1).long()
#
#         return combined_masks
#
#
#     def __call__(self, image, mask):
#         # Step 1: Apply margins to masks
#         # mask = self.__apply_erode_margins(mask)
#         # Step 4: Apply input augmentations
#         image[:,0:3,:,:] = self.input_augs(image[:,0:3,:,:])
#
#         # # Step 2: Apply piecewise affine transformation
#         # image, mask = self.__apply_piecewise_affine(image, mask)
#
#         # Step 3: Apply shape augmentations using RandomOneOf
#         # Add the temporary tensor of ones to track padding
#         ones_tensor = torch.ones_like(image)
#
#         combined = torch.cat([image, mask.unsqueeze(1).float(), ones_tensor], dim=1)
#
#         # Apply the augmentations to the combined tensor (image + mask + ones)
#         combined_augmented = self.shape_augs(combined)
#
#         # After augmentation, split the tensor back into the image, the mask, and the ones tensor
#         image = combined_augmented[:, :image.size(1)]  # Only the image part
#         mask = combined_augmented[:, image.size(1):image.size(1) + mask.unsqueeze(1).size(1)]  # Only the mask part
#         ones_aug = combined_augmented[:, image.size(1) + mask.size(1):]  # Only the ones tensor
#
#         # Set padding areas of the image to 1 (based on ones_aug tracking)
#         padding_mask = ones_aug == 0  # This will be True in padding areas
#         image = torch.where(padding_mask, torch.tensor(1.0, device=image.device), image)
#         #
#         # # Step 4: Apply input augmentations
#         # image[:,0:3,:,:] = self.input_augs(image[:,0:3,:,:])
#         #
#         # # Step 5: Combine augmented mask with the image
#         # # image1 = (1 - mask) + image
#         #
#         # # Step 6: Resize to final size
#         mask = mask.squeeze(1)
#         return image, mask


def Data_class_analyze(masks,# a numpy array (batch, H, W, C)
                       class_labels): # A list of classes with correct order. for example class0 should be placed in first place.
    """This function analyzes the images and masks. it counts number and area of samples for each class.
    It is a good representation for imbalance data. It also gives hints to data augmentation and over-sampling"""

    shape = np.shape(masks)
    num_classes = len(class_labels)
    class_samples = np.zeros(num_classes)
    class_areas = np.zeros(num_classes)
    class_distribution = np.zeros((shape[0],num_classes))
    for i in range(shape[0]):
        msk = masks[i]
        if len(msk.shape)>2:
            msk = np.sum(msk, axis=2)# convert to one image
        for j in range(num_classes):
            area = np.sum(msk == j)
            if area > 0:
                # if j == 5:
                #     plt.imshow(msk)
                #     plt.show()
                class_samples[j] += 1
                class_areas[j] += area
                class_distribution[i,j] += 1
    return class_samples, class_areas, class_distribution



def split_train_val(class_samples = 0, class_distribution = 0, val_percent = 0.2):
    a = 0
    class_distribution1 = np.copy(class_distribution)
    shape = np.shape(class_distribution)
    val_samples = int(shape[0]*val_percent)
    train_samples = shape[0] - val_samples
    val_index = np.zeros(val_samples)
    train_index = np.zeros(train_samples)
    val_indexes = 0
    val_samples1 = np.copy(val_samples)
    for i in range(1,shape[1]):
        ind = np.where(class_samples[1:] == np.min(class_samples[1:]))[0]+1
        ind = ind[0]
        val = class_samples[ind]
        class_samples[ind] = np.inf
        random.seed(42)
        val_sample_inds = random.sample(range(int(val)), int(np.ceil(val*val_percent)))
        print(val_sample_inds)
        k = 0
        for m in range(shape[0]):
            if class_distribution[m,int(ind)] == 1:
                if len(np.where(np.array(val_sample_inds) == k)[0]):
                    val_index[val_indexes] = m
                    class_distribution[m] = 0*class_distribution[m]
                    val_indexes += 1
                    val_samples1 -= 1
                    if val_samples1 == 0:
                        break
                    # print(k)
                k += 1
        if val_samples1 ==0:
            break
    val_index = np.where(np.sum(class_distribution,axis = 1) == 0)
    train_index = np.where(np.sum(class_distribution, axis=1) != 0)

    # print(val_index)





    return train_index, val_index


def addsamples(images,mask,sample_th = 0.2, tissue_labels = None):
    a = 2
    angles = [45,90,135,180,225,270,325]
    shape = np.shape(images)
    shape_m = np.shape(mask)
    new_ims = []
    new_masks = []
    class_samples, class_areas, class_distribution = Data_class_analyze(mask, tissue_labels)
    average_class = class_samples / np.shape(mask)[0]
    average_area = class_areas / np.sum(class_areas)
    avg = 100 * average_class * 100 * average_area / np.sum(100 * average_class * 100 * average_area)
    new_ims = []
    new_masks = []
    for i in range(1,shape_m[3]):# start from 1 to skip background
        ind = np.where(avg[1:] == np.min(avg[1:]))[0]+1
        ind = ind[0]
        val = avg[ind]
        avg[ind] = np.inf
        class_inds = np.where(class_distribution[:,ind] == 1)
        if val < sample_th:
            up_num = int(shape[0] * sample_th/(1-sample_th) - 100*average_area[ind])
            for j in range(3):# fliplr, flipud
                if up_num <= 0:
                    break
                for inds in class_inds[0]:
                    if up_num <= 0:
                        break
                    new_ims.append(cv2.flip(images[inds],j-1))
                    new_masks.append(cv2.flip(mask[inds],j-1))
                    up_num -= 1

            for rot in angles:
                if up_num <= 0:
                    break
                for inds in class_inds[0]:
                    if up_num <= 0:
                        break

                    M = cv2.getRotationMatrix2D((shape[1] / 2, shape[2] / 2), rot, 1)
                    imm = cv2.warpAffine(images[inds], M, (shape[1], shape[2]),borderValue=(255,255,255))
                    new_ims.append(imm)
                    imm = cv2.warpAffine(np.array(mask[inds], dtype=np.uint8), M, (shape[1], shape[2]))
                    new_masks.append(imm)


                    up_num -= 1

            for j in range(3):  # fliplr, flipud
                if up_num <= 0:
                    break
                for inds in class_inds[0]:
                    if up_num <= 0:
                        break

                    image = cv2.flip(images[inds], j-1)
                    msk = cv2.flip(mask[inds], j-1)
                    for rot in angles:
                        if up_num <= 0:
                            break
                        M = cv2.getRotationMatrix2D((shape[1] / 2, shape[2] / 2), rot, 1)
                        imm = cv2.warpAffine(image, M, (shape[1], shape[2]),borderValue=(255,255,255))
                        new_ims.append(imm)
                        imm = cv2.warpAffine(np.array(msk, dtype=np.uint8), M, (shape[1], shape[2]))
                        new_masks.append(imm)
                        up_num -= 1


            class_distribution[class_inds] = 0 * class_distribution[class_inds]
    a = 0
    images1 = np.zeros((shape[0] + len(new_ims), shape[1], shape[2], shape[3]))
    mask1 = np.zeros((shape[0] + len(new_ims), shape[1], shape[2], shape_m[3]))
    images1[0:shape[0]] = images
    mask1[0:shape[0]] = mask

    new_ims1 = np.array(new_ims)
    new_masks1 = np.array(new_masks, dtype=np.uint8)
    images1[shape[0]:] = new_ims1
    mask1[shape[0]:] = new_masks1
    images1 = images1.astype('float32')
    mask1 = mask1.astype('int64')

    return images1, mask1



#
# def puma_dice_loss(preds, targets, eps=1e-6):
#     """
#     Compute the Dice loss for binary or multi-class segmentation.
#
#     Args:
#         preds (torch.Tensor): Predicted tensor of shape (B, C, H, W).
#         targets (torch.Tensor): Ground truth tensor of shape (B, H, W).
#         eps (float): Small epsilon to avoid division by zero.
#
#     Returns:
#         torch.Tensor: Dice loss value.
#     """
#     num_classes = preds.size(1)
#     targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
#
#     intersection = torch.sum(preds * targets_one_hot, dim=(2, 3))
#     union = torch.sum(preds + targets_one_hot, dim=(2, 3))
#
#     dice = (2 * intersection + eps) / (union + eps)
#     return 1 - dice.mean()  # Dice loss is 1 - Dice score


# def compute_puma_dice_micro_dice(model = None, target_siz = None,epoch = 1, input_folder = '', output_folder = '', ground_truth_folder = '', device = None, model1=None, weights_list = None):
#     # input_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_images2"
#     # output_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_prediction2"
#     if device == None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     if epoch == 1:
#         if os.path.exists(output_folder):
#             for root, dirs, files in os.walk(output_folder, topdown=False):
#                 for file in files:
#                     os.remove(os.path.join(root, file))
#                 for dir in dirs:
#                     os.rmdir(os.path.join(root, dir))
#             os.rmdir(output_folder)
#         os.makedirs(output_folder, exist_ok=True)
#
#     # Create the output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
#
#     # Define preprocessing transforms
#
#     # Process each image
#     for file_name in os.listdir(input_folder):
#         if file_name.endswith(".tif"):
#             input_path = os.path.join(input_folder, file_name)
#             output_path = os.path.join(output_folder, file_name)
#
#             # Read the TIF image
#             image = tifffile.imread(input_path)
#
#             # Ensure the image has 3 channels
#             # Handle 4-channel images by dropping the alpha channel
#             if image.shape[2] == 4:
#                 image = image[:, :, :3]  # Keep only the first three channels (RGB)
#             elif image.shape[2] != 3:
#                 raise ValueError(f"Unexpected number of channels in image: {file_name}")
#             image = cv2.resize(image, target_siz)
#
#             image = image / 255
#
#             image = np.transpose(image, (2, 0, 1))
#             image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
#
#             # image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
#             # val_outputs1 = sliding_window_inference(image_tensor, roi_size, sw_batch_size, model)
#             # val_outputs1 = [post_trans(i) for i in decollate_batch(val_outputs1)]
#
#             # Get prediction
#             if weights_list != None:
#                 for k in range(len(weights_list)):
#                     model.load_state_dict(torch.load(weights_list[k], weights_only=True))
#                     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#                     model.to(device)
#                     model.eval()
#                     if k == 0:
#                         prediction = F.softmax(model(image_tensor), dim=1)
#
#                     else:
#                         prediction = prediction + F.softmax(model(image_tensor), dim=1)
#                 prediction = prediction / len(weights_list)
#                 # prediction = F.softmax(prediction, dim=1)
#             else:
#                 prediction = model(image_tensor)
#                 prediction = F.softmax(prediction, dim=1)
#                 if model1 is not None:
#                     prediction1 = model1(image_tensor)
#                     prediction1 = F.softmax(prediction1, dim=1)
#
#                     prediction = 0.5*prediction + 0.5*prediction1
#
#                 # Post-process prediction (e.g., apply softmax or argmax)
#             prediction = torch.argmax(prediction[0], dim=0).squeeze(0).cpu().numpy()
#             prediction[prediction>5] = prediction[prediction>5]
#
#             # Save the prediction as a TIF file
#             with tifffile.TiffWriter(output_path) as tif:
#                 tif.write(prediction.astype(np.uint8), resolution=(300, 300))
#
#             # print(f"Processed and saved: {file_name}")
#     # ground_truth_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_ground_truth2"
#     prediction_folder = output_folder #"/home/ntorbati/PycharmProjects/pythonProject/validation_prediction2"
#     image_shape = (1024, 1024)  # Adjust based on your images' resolution
#
#     dice_scores, mean_puma_dice, mean_dice_classes = compute_dice_scores(ground_truth_folder, prediction_folder,
#                                                                          image_shape)
#     # print("Overall Dice Scores:", mean_puma_dice)
#     # print("Overall Mean Dice Scores:", mean_dice_classes)
#
#     # for file, scores in dice_scores.items():
#     #     print(f"{file}: {scores}")
#
#
#     micro_dices, mean_micro_dice = calculate_micro_dice_score_with_masks(ground_truth_folder, prediction_folder,
#                                                                          image_shape, eps=0.00001)
#
#
#     return mean_puma_dice, micro_dices, mean_micro_dice



def puma_f1_loss(preds, targets, eps=1e-6, radius=15):
    """
    Compute the F1 loss for segmentation tasks using KD-Tree for efficient nearest neighbor search.

    Args:
        preds (torch.Tensor): Predicted tensor of shape (B, C, H, W).
        targets (torch.Tensor): Ground truth tensor of shape (B, H, W).
        eps (float): Small epsilon to avoid division by zero.
        radius (int): The radius within which predictions are considered valid.

    Returns:
        torch.Tensor: F1 loss value.
    """
    num_classes = preds.size(1)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    preds = F.softmax(preds, dim=1) if num_classes > 1 else torch.sigmoid(preds)

    preds_binary = (preds > 0.5).float()  # Thresholding predictions

    TP = torch.zeros(1, device=preds.device)
    FP = torch.zeros(1, device=preds.device)
    FN = torch.zeros(1, device=preds.device)

    for b in range(preds.shape[0]):  # Iterate over batch
        for c in range(1,num_classes):  # Iterate over classes
            pred_mask = preds_binary[b, c]
            gt_mask = targets_one_hot[b, c]

            if pred_mask.sum() == 0 and gt_mask.sum() == 0:
                continue  # No objects, skip

            pred_coords = pred_mask.nonzero().cpu().numpy()
            gt_coords = gt_mask.nonzero().cpu().numpy()

            if len(gt_coords) > 0:
                tree = KDTree(gt_coords)
                distances, indices = tree.query(pred_coords)
                matched = distances < radius

                matched_indices = set()
                for i, (match, idx) in enumerate(zip(matched, indices)):
                    if match and idx not in matched_indices:
                        TP += 1
                        matched_indices.add(idx)
                    else:
                        FP += 1

                FN += len(gt_coords) - len(matched_indices)
                if FN < 0:
                    print('wtf')
            else:
                FP += len(pred_coords)

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1_score = (2 * precision * recall) / (precision + recall + eps)

    return 1 - f1_score.mean()


def f1_custom(gt, pred):
    tp = np.count_nonzero(gt * pred)
    fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(gt & ~pred)
    f1 = (2 * tp) / ((2 * tp) + fp + fn + 1e-6)
    return f1


def puma_f1_loss_custom(preds, targets, eps=1e-6, radius=15, f1_ret = False,remove_bg = 0):
    """
    Compute the F1 loss for segmentation tasks using KD-Tree for efficient nearest neighbor search.

    Uses Dice loss for the background (class 0) and F1 loss for other classes.

    Args:
        preds (torch.Tensor): Predicted tensor of shape (B, C, H, W).
        targets (torch.Tensor): Ground truth tensor of shape (B, H, W).
        eps (float): Small epsilon to avoid division by zero.
        radius (int): The radius within which predictions are considered valid.

    Returns:
        torch.Tensor: Combined loss value.
    """
    num_classes = preds.size(1)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    preds_binary = (preds > 0.5).float()  # Thresholding predictions

    loss = torch.tensor(0.0, device=preds.device)
    f1 = torch.zeros(num_classes, device=preds.device)
    # f1_b = torch.zeros(num_classes, device=preds.device)

    for b in range(preds.shape[0]):  # Iterate over batch
        for c in range(num_classes):  # Iterate over classes
            pred_mask = preds_binary[b, c].cpu().numpy().astype(bool)
            gt_mask = targets_one_hot[b, c].cpu().numpy().astype(bool)
            # print(f1_custom(gt_mask, pred_mask))
            # if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
            #     loss += 0
            #     f1[c] = 1
            # else:
            loss += 1 - f1_custom(gt_mask, pred_mask)
            f1[c] += f1_custom(gt_mask, pred_mask)
        # print(f1)
        # f1_b += f1
    if f1_ret:
        return f1[remove_bg:].mean()
    else:
        return loss / num_classes


def circular_augmentation(train_images, masks, target_class, r1, r2, d1):
    """
    Apply circular augmentation to the specified class in the segmentation mask.

    Parameters:
        train_images (torch.Tensor): Input tensor of training images, size (B, C, H, W).
        masks (torch.Tensor): Input tensor of mask images, size (B, H, W) (torch.long).
        target_class (int): The class on which to apply the augmentation.
        r1 (int): Minimum radius of circles.
        r2 (int): Maximum radius of circles.
        d1 (float): Density of circles (fraction of target class area to be covered).

    Returns:
        tuple: Augmented training images and masks.
    """
    # Get device from input tensors
    device = train_images.device

    # Get dimensions
    B, C, H, W = train_images.shape
    augmented_images = train_images.clone()
    augmented_masks = masks.clone()

    for b in range(B):
        # Extract the target class region from the mask
        target_region = (masks[b] == target_class)
        target_area = target_region.sum().item()

        if target_area == 0:
            continue  # Skip if the target class is not present

        # Calculate maximum allowable circle area
        max_circle_area = target_area * d1
        current_area = 0

        circles = []  # Track placed circles as (x, y, r)

        while current_area < max_circle_area:
            # Random radius
            r = np.random.randint(r1, r2 + 1)

            # Generate a random center within the valid target region
            valid_y, valid_x = torch.where(target_region)
            if len(valid_y) == 0:
                break  # Exit if no valid points are left
            idx = np.random.randint(len(valid_y))
            x, y = valid_x[idx].item(), valid_y[idx].item()

            # Check for overlap
            overlap = False
            for cx, cy, cr in circles:
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist < r + cr:
                    overlap = True
                    break

            if not overlap:
                # Generate a circular area
                yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                circle_area = ((xx - x) ** 2 + (yy - y) ** 2) <= r ** 2

                # Ensure the circle is within the target region
                circle_mask = circle_area & target_region

                new_area = circle_mask.sum().item()
                # if current_area + new_area <= max_circle_area:
                    # Apply the augmentation: set the mask to 0 and the image to 0 in the circle
                augmented_masks[b][circle_mask] = 0
                augmented_images[b][:, circle_mask] = 1

                current_area += new_area
                circles.append((x, y, r))

    return augmented_images, augmented_masks

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = torch.nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def upsample_necro(image_data = None,mask_data = None):
    rows, cols = np.shape(image_data)[1], np.shape(image_data)[2]  # Replace with your image dimensions
    r, c = np.indices((rows, cols))

    # Create masks
    upper_diag = c >= r
    lower_diag = c <= r
    upper_anti_diag = r + c < rows
    lower_anti_diag = r + c >= rows - 1
    # Top half mask
    top_half = r < rows // 2

    # Left half mask
    left_half = c < cols // 2
    masks = [upper_diag, lower_diag, upper_anti_diag, lower_anti_diag,top_half, left_half]
    masks_inds = [
        ]
    for j in range(mask_data.shape[0]):
        for i in [5]:
            area = np.sum(mask_data[j, :, :] == i)
            # print(area)
            if area > 0:
                if image_data is not None:
                    # plt.subplot(1, 2, 1)
                    # plt.imshow(image_data[j] / 255)
                    # plt.subplot(1, 2, 2)
                    # plt.imshow(mask_data[j])
                    # plt.show()
                    if j == 7:
                        msk = upper_anti_diag
                        im_new = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_new[:,:,kk] = image_data[j,:,:,kk]*msk
                            im_new[:,:,kk] = im_new[:,:,kk] + 255*(1-msk)
                        msk_new = mask_data[j]*msk

                        im_old = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_old[:,:,kk] = image_data[j,:,:,kk]*(1-msk)
                            im_old[:,:,kk] = im_old[:,:,kk] + 255*msk
                        msk_old = mask_data[j]*(1-msk)
                        image_data[j] = im_old
                        mask_data[j] = msk_old

                        new_data = im_new[np.newaxis,:]
                        new_mask = msk_new[np.newaxis,:]
                    if j == 51:
                        msk = left_half
                        im_new = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_new[:,:,kk] = image_data[j,:,:,kk]*msk
                            im_new[:,:,kk] = im_new[:,:,kk] + 255*(1-msk)
                        msk_new = mask_data[j]*msk

                        im_old = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_old[:,:,kk] = image_data[j,:,:,kk]*(1-msk)
                            im_old[:,:,kk] = im_old[:,:,kk] + 255*msk
                        msk_old = mask_data[j]*(1-msk)
                        image_data[j] = im_old
                        mask_data[j] = msk_old

                        new_data = np.concatenate((im_new[np.newaxis, :], new_data),axis=0)
                        new_mask = np.concatenate((msk_new[np.newaxis, :], new_mask),axis=0)

                    if j == 100:
                        msk = upper_diag
                        im_new = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_new[:,:,kk] = image_data[j,:,:,kk]*msk
                            im_new[:,:,kk] = im_new[:,:,kk] + 255*(1-msk)
                        msk_new = mask_data[j]*msk

                        im_old = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_old[:,:,kk] = image_data[j,:,:,kk]*(1-msk)
                            im_old[:,:,kk] = im_old[:,:,kk] + 255*msk
                        msk_old = mask_data[j]*(1-msk)
                        image_data[j] = im_old
                        mask_data[j] = msk_old

                        new_data = np.concatenate((im_new[np.newaxis, :], new_data),axis=0)
                        new_mask = np.concatenate((msk_new[np.newaxis, :], new_mask),axis=0)

                    if j == 101:
                        msk = upper_diag
                        im_new = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_new[:,:,kk] = image_data[j,:,:,kk]*msk
                            im_new[:,:,kk] = im_new[:,:,kk] + 255*(1-msk)
                        msk_new = mask_data[j]*msk

                        im_old = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_old[:,:,kk] = image_data[j,:,:,kk]*(1-msk)
                            im_old[:,:,kk] = im_old[:,:,kk] + 255*msk
                        msk_old = mask_data[j]*(1-msk)
                        image_data[j] = im_old
                        mask_data[j] = msk_old

                        new_data = np.concatenate((im_new[np.newaxis, :], new_data),axis=0)
                        new_mask = np.concatenate((msk_new[np.newaxis, :], new_mask),axis=0)


                    if j == 102:
                        msk = upper_diag
                        im_new = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_new[:,:,kk] = image_data[j,:,:,kk]*msk
                            im_new[:,:,kk] = im_new[:,:,kk] + 255*(1-msk)
                        msk_new = mask_data[j]*msk

                        im_old = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_old[:,:,kk] = image_data[j,:,:,kk]*(1-msk)
                            im_old[:,:,kk] = im_old[:,:,kk] + 255*msk
                        msk_old = mask_data[j]*(1-msk)
                        image_data[j] = im_old
                        mask_data[j] = msk_old

                        new_data = np.concatenate((im_new[np.newaxis, :], new_data),axis=0)
                        new_mask = np.concatenate((msk_new[np.newaxis, :], new_mask),axis=0)


                    if j == 127:
                        msk = upper_diag
                        im_new = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_new[:,:,kk] = image_data[j,:,:,kk]*msk
                            im_new[:,:,kk] = im_new[:,:,kk] + 255*(1-msk)
                        msk_new = mask_data[j]*msk

                        im_old = np.zeros_like(image_data[j])
                        for kk in range(image_data.shape[3]):
                            im_old[:,:,kk] = image_data[j,:,:,kk]*(1-msk)
                            im_old[:,:,kk] = im_old[:,:,kk] + 255*msk
                        msk_old = mask_data[j]*(1-msk)
                        image_data[j] = im_old
                        mask_data[j] = msk_old

                        new_data = np.concatenate((im_new[np.newaxis, :], new_data),axis=0)
                        new_mask = np.concatenate((msk_new[np.newaxis, :], new_mask),axis=0)
    image_data = np.concatenate((image_data,new_data),axis=0)
    mask_data = np.concatenate((mask_data,new_mask),axis=0)
    return image_data,mask_data

def Mine_resize(image = None, mask = None, final_size = None):
    """Resize image and mask to the final size."""
    image_resized = F.interpolate(image, size=(final_size[0], final_size[1]), mode="bilinear",
                                  align_corners=False)
    mask_resized_binary = F.interpolate(mask[:,0:2].float(), size=final_size, mode="nearest").long()
    mask_resized_hv = F.interpolate(mask[:,2:], size=(final_size[0], final_size[1]), mode="bilinear",
                                  align_corners=False)
    mask_resized = torch.concatenate((mask_resized_binary,mask_resized_hv), dim=1)
    return image_resized, mask_resized






class RandomOneOf(torch.nn.Module):
    def __init__(self, augmentations, p=1.0):
        super().__init__()
        self.augmentations = torch.nn.ModuleList(augmentations)
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() > self.p:
            return x  # No augmentation applied
        # Randomly select one augmentation
        idx = torch.randint(0, len(self.augmentations), (1,)).item()
        return self.augmentations[idx](x)


class KorniaAugmentation:
    def __init__(self, mode="train", num_classes=6, seed=None, size = None, regression = False):
        self.mode = mode
        self.size = size
        self.num_classes = num_classes
        self.seed = seed
        self.regression = regression
        torch.manual_seed(seed) if seed else None

        # Define PiecewiseAffine augmentation
        self.piecewise_affine = K.RandomThinPlateSpline(scale=0.1, align_corners=False)

        # Define Shape Transformations
        # self.affine = T.Compose([
        #     T.RandomAffine(
        #         degrees=(-179, 179),  # Rotation between -179 and 179 degrees
        #         translate=(0.01, 0.01),  # Translation up to 1% of image dimensions
        #         scale=(0.8, 1.2),  # Scaling between 0.8 and 1.2
        #         shear=(-5, 5),  # Shear range between -5 and 5 degrees
        #         # interpolation=InterpolationMode.NEAREST,  # Use nearest neighbor (0 corresponds to nearest)
        #         fill=0  # Fill padding areas with black (0)
        #     )
        # ])

        # Create a RandomOneOf augmentation for the shape transformations
        self.shape_augs = torch.nn.Sequential(#RandomOneOf([
            # K.RandomThinPlateSpline(p=0.5, scale=0.1, same_on_batch=True, keepdim=True),  # Correct parameters
            T.RandomAffine(
                degrees=(-179, 179),  # Rotation between -179 and 179 degrees
                translate=(0.1, 0.1),  # Translation up to 1% of image dimensions
                scale=(0.9, 1.1),  # Scaling between 0.8 and 1.2
                shear=(-5, 5),  # Shear range between -5 and 5 degrees
                # interpolation=InterpolationMode.NEAREST,  # Use nearest neighbor (0 corresponds to nearest)
                fill=0  # Fill padding areas with black (0)
            ),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            # K.CenterCrop(size=size)
        )

        # Define Input Transformations
        self.input_augs = torch.nn.Sequential(
            RandomOneOf([
                K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
                K.RandomMedianBlur((3, 3), p=1.0),
                K.RandomGaussianNoise(mean=0.0, std=0.05, p=1.0),
            ], p=1.0),
            K.RandomBrightness(brightness=(0.9, 1.1), p=1.0),
            K.RandomContrast(contrast=(0.75, 1.25), p=1.0),
            K.RandomHue(hue=(-0.05, 0.05), p=1.0),
            K.RandomSaturation(saturation=(0.8, 1.2), p=1.0),

        )

    def __apply_piecewise_affine(self, images, masks):
        """Apply piecewise affine transformation to both images and masks."""
        # Stack masks into additional channels

        B, C, H, W = images.shape
        masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Create a temporary ones tensor to track padding
        ones_tensor = torch.ones_like(images)

        # Combine images, masks, and the ones tensor for padding tracking
        combined = torch.cat([images, masks_one_hot, ones_tensor], dim=1)

        # Apply affine transformation to the combined tensor (image + mask + ones)
        combined_augmented = self.piecewise_affine(combined)

        # Separate images, masks, and ones tensor after transformation
        images_aug = combined_augmented[:, :C]
        masks_aug = combined_augmented[:, C:C + self.num_classes]
        ones_aug = combined_augmented[:, C + self.num_classes:]


        masks_aug = masks_aug.argmax(dim=1).long()  # Convert back to class labels

        # Track padding by checking where ones_aug has been turned to 0 (padding areas)
        padding_mask = ones_aug == 0  # This will be True in the padding areas

        # Fill padding for images with 1 (where padding_mask is True)
        images_aug = torch.where(padding_mask, torch.tensor(1.0, device=images_aug.device), images_aug)
        return images_aug, masks_aug

    def __apply_erode_margins(self, masks):
        """Add margins to masks by applying erosion."""
        B, H, W = masks.shape
        masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Apply erosion to each class
        kernel = torch.ones((3, 3), device=masks.device)
        eroded_channels = []
        for i in range(self.num_classes):
            mask_channel = masks_one_hot[:, i:i + 1]
            eroded = mask_channel
            for _ in range(5):  # Perform erosion 3 times
                eroded = KM.erosion(eroded, kernel)
            eroded_channels.append(eroded)

        # Combine eroded masks
        eroded_masks = torch.cat(eroded_channels, dim=1)
        combined_masks = eroded_masks.argmax(dim=1).long()

        return combined_masks


    def __call__(self, image, mask):
        # Step 1: Apply margins to masks
        # mask = self.__apply_erode_margins(mask)
        # Step 4: Apply input augmentations
        image[:,0:3,:,:] = self.input_augs(image[:,0:3,:,:].contiguous())

        # # Step 2: Apply piecewise affine transformation
        # image, mask = self.__apply_piecewise_affine(image, mask)

        # Step 3: Apply shape augmentations using RandomOneOf
        # Add the temporary tensor of ones to track padding
        ones_tensor = torch.ones_like(image)


        if len(mask.shape) == 4:
            combined = torch.cat([image, mask.float(), ones_tensor], dim=1)
        else:
            combined = torch.cat([image, mask.unsqueeze(1).float(), ones_tensor], dim=1)

        # Apply the augmentations to the combined tensor (image + mask + ones)
        combined_augmented = self.shape_augs(combined)

        # After augmentation, split the tensor back into the image, the mask, and the ones tensor
        image = combined_augmented[:, :image.size(1)]  # Only the image part
        if len(mask.shape) == 4:
            mask = combined_augmented[:, image.size(1):image.size(1) + mask.size(1)]  # Only the mask part
        else:
            mask = combined_augmented[:, image.size(1):image.size(1) + mask.unsqueeze(1).size(1)]  # Only the mask part
        ones_aug = combined_augmented[:, image.size(1) + mask.size(1):]  # Only the ones tensor

        # Set padding areas of the image to 1 (based on ones_aug tracking)
        padding_mask = ones_aug == 0  # This will be True in padding areas
        image[:,0:3] = torch.where(padding_mask[:,0:3], torch.tensor(1.0, device=image.device), image[:,0:3])

        if self.regression:
            mask[:,0] = torch.where(padding_mask[:,0], torch.tensor(1.0, device=image.device), mask[:,0])
        #
        # # Step 4: Apply input augmentations
        # image[:,0:3,:,:] = self.input_augs(image[:,0:3,:,:])
        #
        # # Step 5: Combine augmented mask with the image
        # # image1 = (1 - mask) + image
        #
        # # Step 6: Resize to final size
        if not self.regression:
            mask = mask.squeeze(1)
        return image, mask


def Data_class_analyze(masks,# a numpy array (batch, H, W, C)
                       class_labels): # A list of classes with correct order. for example class0 should be placed in first place.
    """This function analyzes the images and masks. it counts number and area of samples for each class.
    It is a good representation for imbalance data. It also gives hints to data augmentation and over-sampling"""

    shape = np.shape(masks)
    num_classes = len(class_labels)
    class_samples = np.zeros(num_classes)
    class_areas = np.zeros(num_classes)
    class_distribution = np.zeros((shape[0],num_classes))
    for i in range(shape[0]):
        msk = masks[i]
        if len(msk.shape)>2:
            msk = np.sum(msk, axis=2)# convert to one image
        for j in range(num_classes):
            area = np.sum(msk == j)
            if area > 0:
                # if j == 5:
                #     plt.imshow(msk)
                #     plt.show()
                class_samples[j] += 1
                class_areas[j] += area
                class_distribution[i,j] += 1
    return class_samples, class_areas, class_distribution



def split_train_val(class_samples = 0, class_distribution = 0, val_percent = 0.2):
    a = 0
    class_distribution1 = np.copy(class_distribution)
    shape = np.shape(class_distribution)
    val_samples = int(shape[0]*val_percent)
    train_samples = shape[0] - val_samples
    val_index = np.zeros(val_samples)
    train_index = np.zeros(train_samples)
    val_indexes = 0
    val_samples1 = np.copy(val_samples)
    for i in range(1,shape[1]):
        ind = np.where(class_samples[1:] == np.min(class_samples[1:]))[0]+1
        ind = ind[0]
        val = class_samples[ind]
        class_samples[ind] = np.inf
        random.seed(42)
        val_sample_inds = random.sample(range(int(val)), int(np.ceil(val*val_percent)))
        print(val_sample_inds)
        k = 0
        for m in range(shape[0]):
            if class_distribution[m,int(ind)] == 1:
                if len(np.where(np.array(val_sample_inds) == k)[0]):
                    val_index[val_indexes] = m
                    class_distribution[m] = 0*class_distribution[m]
                    val_indexes += 1
                    val_samples1 -= 1
                    if val_samples1 == 0:
                        break
                    # print(k)
                k += 1
        if val_samples1 ==0:
            break
    val_index = np.where(np.sum(class_distribution,axis = 1) == 0)
    train_index = np.where(np.sum(class_distribution, axis=1) != 0)

    # print(val_index)





    return train_index, val_index


def addsamples(images,mask,sample_th = 0.2, tissue_labels = None):
    a = 2
    angles = [45,90,135,180,225,270,325]
    shape = np.shape(images)
    shape_m = np.shape(mask)
    new_ims = []
    new_masks = []
    class_samples, class_areas, class_distribution = Data_class_analyze(mask, tissue_labels)
    average_class = class_samples / np.shape(mask)[0]
    average_area = class_areas / np.sum(class_areas)
    avg = 100 * average_class * 100 * average_area / np.sum(100 * average_class * 100 * average_area)
    new_ims = []
    new_masks = []
    for i in range(1,shape_m[3]):# start from 1 to skip background
        ind = np.where(avg[1:] == np.min(avg[1:]))[0]+1
        ind = ind[0]
        val = avg[ind]
        avg[ind] = np.inf
        class_inds = np.where(class_distribution[:,ind] == 1)
        if val < sample_th:
            up_num = int(shape[0] * sample_th/(1-sample_th) - 100*average_area[ind])
            for j in range(3):# fliplr, flipud
                if up_num <= 0:
                    break
                for inds in class_inds[0]:
                    if up_num <= 0:
                        break
                    new_ims.append(cv2.flip(images[inds],j-1))
                    new_masks.append(cv2.flip(mask[inds],j-1))
                    up_num -= 1

            for rot in angles:
                if up_num <= 0:
                    break
                for inds in class_inds[0]:
                    if up_num <= 0:
                        break

                    M = cv2.getRotationMatrix2D((shape[1] / 2, shape[2] / 2), rot, 1)
                    imm = cv2.warpAffine(images[inds], M, (shape[1], shape[2]),borderValue=(255,255,255))
                    new_ims.append(imm)
                    imm = cv2.warpAffine(np.array(mask[inds], dtype=np.uint8), M, (shape[1], shape[2]))
                    new_masks.append(imm)


                    up_num -= 1

            for j in range(3):  # fliplr, flipud
                if up_num <= 0:
                    break
                for inds in class_inds[0]:
                    if up_num <= 0:
                        break

                    image = cv2.flip(images[inds], j-1)
                    msk = cv2.flip(mask[inds], j-1)
                    for rot in angles:
                        if up_num <= 0:
                            break
                        M = cv2.getRotationMatrix2D((shape[1] / 2, shape[2] / 2), rot, 1)
                        imm = cv2.warpAffine(image, M, (shape[1], shape[2]),borderValue=(255,255,255))
                        new_ims.append(imm)
                        imm = cv2.warpAffine(np.array(msk, dtype=np.uint8), M, (shape[1], shape[2]))
                        new_masks.append(imm)
                        up_num -= 1


            class_distribution[class_inds] = 0 * class_distribution[class_inds]
    a = 0
    images1 = np.zeros((shape[0] + len(new_ims), shape[1], shape[2], shape[3]))
    mask1 = np.zeros((shape[0] + len(new_ims), shape[1], shape[2], shape_m[3]))
    images1[0:shape[0]] = images
    mask1[0:shape[0]] = mask

    new_ims1 = np.array(new_ims)
    new_masks1 = np.array(new_masks, dtype=np.uint8)
    images1[shape[0]:] = new_ims1
    mask1[shape[0]:] = new_masks1
    images1 = images1.astype('float32')
    mask1 = mask1.astype('int64')

    return images1, mask1





def dice_loss_binary(preds, targets, eps=1e-5):
    # preds: [B, 1, H, W] logits
    # targets: [B, H, W] with {0,1}

    preds = torch.sigmoid(preds)
    targets = targets.float().unsqueeze(1)  # [B, 1, H, W]

    intersection = torch.sum(preds * targets, dim=(2, 3))
    union = torch.sum(preds + targets, dim=(2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def dilate_erode(image = None, disk_radius = 1, itersations = 1, dilate = True):
    kernel_size = (2 * disk_radius + 1, 2 * disk_radius + 1)

    # Create the circular disk structuring element
    disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    # Apply dilation

    if dilate:
        dilated_image = cv2.dilate(np.array(image, dtype=np.uint8), disk_kernel, iterations=itersations)
        return dilated_image
    else:
        eroded_image = cv2.erode(np.array(image, dtype=np.uint8), disk_kernel, iterations=itersations)
        return eroded_image


def get_center_of_gravity(mask):
    """Center of gravity using soft weights, ignoring zeros."""
    mask = np.where(mask > 0, mask, 0)
    h, w = mask.shape
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    total_weight = mask.sum()
    if total_weight == 0:
        return None
    x_center = (x_coords * mask).sum() / total_weight
    y_center = (y_coords * mask).sum() / total_weight
    return (y_center, x_center)

def get_max_activation_point(mask):
    """Point with maximum prediction value > 0."""
    if np.max(mask) == 0:
        return None
    masked = np.where(mask > 0, mask, -np.inf)
    return np.unravel_index(np.argmax(masked), mask.shape)

def get_thresholded_center(mask, threshold=0.5):
    """Center of gravity on a thresholded mask, ignoring zeros."""
    binary_mask = ((mask >= threshold) & (mask > 0)).astype(np.float32)
    return get_center_of_gravity(binary_mask)

def get_bounding_box_center(mask, threshold=0.5):
    """Geometric center of bounding box, ignoring zero values."""
    binary_mask = ((mask >= threshold) & (mask > 0)).astype(np.uint8)
    ys, xs = np.where(binary_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return ((y_min + y_max) / 2, (x_min + x_max) / 2)

def get_topk_weighted_center(mask, k_percent=5):
    """Weighted center using top-k% values, excluding zeros."""
    flat = mask.flatten()
    flat = flat[flat > 0]  # Remove zeros
    if len(flat) == 0:
        return None
    k = max(1, int(len(flat) * k_percent / 100))
    topk_threshold = np.partition(flat, -k)[-k]
    topk_mask = (mask >= topk_threshold) & (mask > 0)
    return get_center_of_gravity(topk_mask.astype(np.float32))

def get_centers(mask):
    c1 =  get_center_of_gravity(mask)
    c2 = get_max_activation_point(mask)
    c3 = get_thresholded_center(mask, 0.5)
    c4 = get_bounding_box_center(mask, 0.5)
    c5 = get_topk_weighted_center(mask, 5)
    return c1, c2, c3, c4, c5


def keep_only_highest_prob(prediction = None):
    prediction[0, 0][prediction[0, 1] > 0.1] = 0
    prediction[0, 0][prediction[0, 2] > 0.1] = 0
    prediction[0, 1][prediction[0, 1] < 0.5] = 0
    prediction[0, 2][prediction[0, 2] < 0.5] = 0
    preds = torch.argmax(prediction.squeeze(0), dim = 0).cpu().numpy()
    full_instance_map1 = merge_instance_maps(preds, preds)
    for i in range(1,3):
        true_pred = prediction[0, i].cpu().numpy()
        pred = np.array(full_instance_map1 == i, dtype=np.uint8)
        true_pred = true_pred*pred
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
        for label in range(1, num_labels):
            mx = np.median(true_pred[labels==label])
            mask = true_pred*(labels == label)
            mask[mask < mx] = 0
            true_pred[labels == label] = 0
            mask = dilate_erode(image=mask, disk_radius=1, itersations=3, dilate=False)
            # moments = cv2.moments(mask)
            # cx = int(moments["m10"] / moments["m00"])
            # cy = int(moments["m01"] / moments["m00"])
            if np.sum(mask)>0:
                # msk = np.copy(true_pred)
                # msk[mask != 1] = 0
                c1, c2, c3, c4, c5 = get_centers(mask)
                if c4 is None:
                    c4 = c1
                cx = int(c4[1])
                cy = int(c4[0])
                mask = cv2.circle(0*mask,(cx,cy),7,1,-1)
                true_pred[mask  == 1] = pred[mask == 1]
        # plt.imshow(true_pred)
        # plt.show()
        prediction[0, i] = torch.tensor(true_pred, dtype=prediction[0, i].dtype, device=prediction[0, i].device)
    return prediction

def compute_puma_dice_micro_dice(model = None, target_siz = None,epoch = 1, input_folder = '', output_folder = '', ground_truth_folder = '', device = None, model1=None,
                                 weights_list = None, er_di = False, augment_all = True, save_jpg = False, file_path = None,
                                 classifier_mode = False
                                 ,nuclei = False,
                                 in_channles = 3,
                                    tissue_path = '',
                                 dataset_name = None
                                 ):
    # input_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_images2"
    # output_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_prediction2"
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if epoch == 1:
    #     if os.path.exists(output_folder):
    #         for root, dirs, files in os.walk(output_folder, topdown=False):
    #             for file in files:
    #                 os.remove(os.path.join(root, file))
    #             for dir in dirs:
    #                 os.rmdir(os.path.join(root, dir))
    #         os.rmdir(output_folder)
    #     os.makedirs(output_folder, exist_ok=True)
    #
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define preprocessing transforms

    # Process each image
    frame_type =[]
    target_path = os.listdir(input_folder) if file_path is None else [file_path]
    for file_name in target_path:
        if file_name.endswith(".tif"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            # Read the TIF image
            image = tifffile.imread(input_path)
            # if dataset_name == 'M':
            #     image1= tile_or_pad1(torch.tensor(image).permute(2,0,1), torch.tensor(np.zeros((image.shape[0],image.shape[1]))), target_size=(1024, 1024),pad_val=255,only_pad=True)
            #     image = image1[0][0].permute(1,2,0).cpu().numpy()
            # Ensure the image has 3 channels
            # Handle 4-channel images by dropping the alpha channel
            # print(input_path)
            if image.shape[2] == 4:
                image = image[:, :, :3]  # Keep only the first three channels (RGB)
            elif image.shape[2] != 3:
                raise ValueError(f"Unexpected number of channels in image: {file_name}")


            if in_channles == 5:
                tis_path = '/home/ntorbati/PycharmProjects/pythonProject/validation_prediction/PrimarytissueFinal/all/' + file_name
                im = cv2.cvtColor(cv2.imread(tis_path), cv2.COLOR_BGR2GRAY)
                image = np.concatenate((image, im[:, :, np.newaxis]*50), axis=2)
                tis_path = '/home/ntorbati/PycharmProjects/pythonProject/validation_prediction/nuclei_10class_all/' + file_name
                im = cv2.cvtColor(cv2.imread(tis_path), cv2.COLOR_BGR2GRAY)
                image = np.concatenate((image, im[:, :, np.newaxis]*50), axis=2)
            elif dataset_name == 'M':
                tis_path = tissue_path + file_name

                # temp = cv2.resize(temp, new_size, interpolation=cv2.INTER_NEAREST)

                if image.shape[0] * image.shape[1] > 0:#128 * 128:
                    im = cv2.cvtColor(cv2.imread(tis_path.replace('/ourMethodResults/', '/ourMethodResults/')),
                                      cv2.COLOR_BGR2GRAY) * 50
                else:
                    im = cv2.cvtColor(cv2.imread(tis_path.replace('/ourMethodResults/', '/ourMethodResults/')),
                                      cv2.COLOR_BGR2GRAY) * 50 * 3.5
                # im1 = tile_or_pad1(torch.tensor(np.zeros((im.shape[0],im.shape[1],3))).permute(2, 0, 1),
                #                      torch.tensor(im), target_size=(1024, 1024),
                #                      pad_val=0,only_pad = True)
                # im = im1[0][1].cpu().numpy()

                image = np.concatenate((image, im[:, :, np.newaxis]), axis=2)

            elif nuclei:
                tis_path = tissue_path + file_name
                im = cv2.cvtColor(cv2.imread(tis_path), cv2.COLOR_BGR2GRAY)
                image = np.concatenate((image, im[:, :, np.newaxis]*50), axis=2)
            elif in_channles == 4:
                tis_path = '/home/ntorbati/PycharmProjects/pythonProject/validation_prediction/nuclei_10class_all/' + file_name
                im = cv2.cvtColor(cv2.imread(tis_path), cv2.COLOR_BGR2GRAY)
                image = np.concatenate((image, im[:, :, np.newaxis]*50), axis=2)



            if er_di:
                disk_radius = 5
                kernel_size = (2 * disk_radius + 1, 2 * disk_radius + 1)

                # Create the circular disk structuring element
                disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
                eroded_image = cv2.erode(image, disk_kernel, iterations=1)

                # Apply dilation
                dilated_image = cv2.dilate(image, disk_kernel, iterations=1)
                image = np.concatenate((image, eroded_image, dilated_image), axis=2)
            if dataset_name != 'M':
                image = cv2.resize(image, target_siz)

            image = image / 255

            image = np.transpose(image, (2, 0, 1))
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

            # image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
            # val_outputs1 = sliding_window_inference(image_tensor, roi_size, sw_batch_size, model)
            # val_outputs1 = [post_trans(i) for i in decollate_batch(val_outputs1)]

            # Get prediction
            if weights_list != None:
                prediction = validate_with_augmentations_and_ensembling(model, image_tensor, weights_list)
                prediction = torch.unsqueeze(prediction, 0)
            elif augment_all:
                prediction = validate_with_augmentations(model, image_tensor)
                # prediction = prediction[0]
            else:
                prediction = model(image_tensor)
                prediction = F.softmax(prediction, dim=1)
                if model1 is not None:
                    prediction1 = model1(image_tensor)
                    prediction1 = F.softmax(prediction1, dim=1)

                    prediction = 0.5*prediction + 0.5*prediction1
            if nuclei == 2:
                # prediction = keep_only_highest_prob(prediction = prediction)
                prediction = torch.argmax(prediction[0], dim=0).squeeze(0).cpu().numpy()
            else:
                prediction = torch.argmax(prediction[0], dim=0).squeeze(0).cpu().numpy()

                # Post-process prediction (e.g., apply softmax or argmax)
            metas_sum = 0
            metas_class = np.empty((0,5))
            primary_sum = 0
            primary_class = np.empty((0,5))
            a_con = np.zeros((5))
            for i in range(6, 11):
                metas_sum = metas_sum + np.sum(prediction == i)
                a_con[i-6] = np.sum(prediction == i)
            metas_class = np.concatenate((metas_class,a_con[np.newaxis,:]), axis=0)

            for i in range(1, 6):
                primary_sum = primary_sum + np.sum(prediction == i)
                a_con[i-1] = np.sum(prediction == i)

            primary_class = np.concatenate((primary_class,a_con[np.newaxis,:]), axis=0)
            if primary_sum > metas_sum:
                frame_type.append([file_name ,'primary', primary_sum, metas_sum])
            else:
                frame_type.append([file_name ,'metas',primary_sum, metas_sum])
            if not nuclei:
                prediction[prediction>5] = prediction[prediction>5] - 5

            colormap = {
                0: [0, 0, 0],  # Black
                1: [255, 0, 0],  # Red
                2: [0, 255, 0],  # Green
                3: [0, 0, 255],  # Blue
                4: [255, 255, 0],  # Yellow
                5: [255, 0, 255],  # Magenta
                6: [0, 125, 125],  # Black
                7: [125, 0, 125],  # Red
                8: [125, 255, 125],  # Green
                9: [125, 255, 255],  # Blue
                10: [255, 255, 255],  # Yellow
            }
            # Create an RGB image by mapping class values to colormap
            rgb_image = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
            # Save the prediction as a TIF file
            with tifffile.TiffWriter(output_path) as tif:
                tif.write(prediction.astype(np.uint8), resolution=(300, 300))

            if save_jpg:
                for class_value, color in colormap.items():
                    rgb_image[prediction == class_value] = color
                cv2.imwrite(output_path[:-4] + '.jpg', rgb_image)

            if file_path is not None:
                return prediction,primary_class,metas_class

            # print(f"Processed and saved: {file_name}")
    # ground_truth_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_ground_truth2"
    prediction_folder = output_folder #"/home/ntorbati/PycharmProjects/pythonProject/validation_prediction2"
    image_shape = (1024, 1024)  # Adjust based on your images' resolution
    with open(output_folder+'/tissue_type.txt', 'w') as file:
        for sublist in frame_type:
            file.write(" ".join(map(str, sublist)) + "\n")
    dice_scores, mean_puma_dice, mean_dice_classes = compute_dice_scores(ground_truth_folder, prediction_folder,
                                                                         image_shape,nuclei)
    np.save(output_folder+'/tissue_type_metas.npy', metas_class)
    np.save(output_folder+'/tissue_type_primary.npy', primary_class)
    # print("Overall Dice Scores:", mean_puma_dice)
    # print("Overall Mean Dice Scores:", mean_dice_classes)

    # for file, scores in dice_scores.items():
    #     print(f"{file}: {scores}")


    micro_dices, mean_micro_dice = calculate_micro_dice_score_with_masks(ground_truth_folder, prediction_folder,
                                                                         image_shape, eps=0.00001,nuclei = nuclei)


    return mean_puma_dice, micro_dices, mean_micro_dice




def compute_puma_dice_micro_dice_eval(model = None, target_siz = None,epoch = 1, input_folder = '', output_folder = '', ground_truth_folder = '', device = None, model1=None,
                                 weights_list = None, er_di = False, augment_all = True, save_jpg = False, file_path = None,
                                 classifier_mode = False
                                 ,nuclei = False,
                                 in_channles = 3,
                                    tissue_path = '',
                                 dataset_name = None
                                 ):
    # input_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_images2"
    # output_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_prediction2"
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if epoch == 1:
    #     if os.path.exists(output_folder):
    #         for root, dirs, files in os.walk(output_folder, topdown=False):
    #             for file in files:
    #                 os.remove(os.path.join(root, file))
    #             for dir in dirs:
    #                 os.rmdir(os.path.join(root, dir))
    #         os.rmdir(output_folder)
    #     os.makedirs(output_folder, exist_ok=True)
    #
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define preprocessing transforms

    # Process each image
    frame_type =[]
    target_path = os.listdir(input_folder) if file_path is None else [file_path]
    for file_name in target_path:
        if file_name.endswith(".tif"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            # Read the TIF image
            image = tifffile.imread(input_path)
            # if dataset_name == 'M':
            #     image1= tile_or_pad1(torch.tensor(image).permute(2,0,1), torch.tensor(np.zeros((image.shape[0],image.shape[1]))), target_size=(1024, 1024),pad_val=255,only_pad=True)
            #     image = image1[0][0].permute(1,2,0).cpu().numpy()
            # Ensure the image has 3 channels
            # Handle 4-channel images by dropping the alpha channel
            # print(input_path)
            if image.shape[2] == 4:
                image = image[:, :, :3]  # Keep only the first three channels (RGB)
            elif image.shape[2] != 3:
                raise ValueError(f"Unexpected number of channels in image: {file_name}")


            if in_channles == 5:
                tis_path = '/home/ntorbati/PycharmProjects/pythonProject/validation_prediction/PrimarytissueFinal/all/' + file_name
                im = cv2.cvtColor(cv2.imread(tis_path), cv2.COLOR_BGR2GRAY)
                image = np.concatenate((image, im[:, :, np.newaxis]*50), axis=2)
                tis_path = '/home/ntorbati/PycharmProjects/pythonProject/validation_prediction/nuclei_10class_all/' + file_name
                im = cv2.cvtColor(cv2.imread(tis_path), cv2.COLOR_BGR2GRAY)
                image = np.concatenate((image, im[:, :, np.newaxis]*50), axis=2)
            elif dataset_name == 'M':
                tis_path = tissue_path + file_name

                # temp = cv2.resize(temp, new_size, interpolation=cv2.INTER_NEAREST)

                if image.shape[0] * image.shape[1] > 0:#128 * 128:
                    im = cv2.cvtColor(cv2.imread(tis_path.replace('/ourMethodResults/', '/ourMethodResults/')),
                                      cv2.COLOR_BGR2GRAY) * 50
                else:
                    im = cv2.cvtColor(cv2.imread(tis_path.replace('/ourMethodResults/', '/ourMethodResults/')),
                                      cv2.COLOR_BGR2GRAY) * 50 * 3.5
                # im1 = tile_or_pad1(torch.tensor(np.zeros((im.shape[0],im.shape[1],3))).permute(2, 0, 1),
                #                      torch.tensor(im), target_size=(1024, 1024),
                #                      pad_val=0,only_pad = True)
                # im = im1[0][1].cpu().numpy()

                image = np.concatenate((image, im[:, :, np.newaxis]), axis=2)

            elif nuclei:
                tis_path = tissue_path + file_name
                im = cv2.cvtColor(cv2.imread(tis_path), cv2.COLOR_BGR2GRAY)
                image = np.concatenate((image, im[:, :, np.newaxis]*50), axis=2)
            elif in_channles == 4:
                tis_path = '/home/ntorbati/PycharmProjects/pythonProject/validation_prediction/nuclei_10class_all/' + file_name
                im = cv2.cvtColor(cv2.imread(tis_path), cv2.COLOR_BGR2GRAY)
                image = np.concatenate((image, im[:, :, np.newaxis]*50), axis=2)



            if er_di:
                disk_radius = 5
                kernel_size = (2 * disk_radius + 1, 2 * disk_radius + 1)

                # Create the circular disk structuring element
                disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
                eroded_image = cv2.erode(image, disk_kernel, iterations=1)

                # Apply dilation
                dilated_image = cv2.dilate(image, disk_kernel, iterations=1)
                image = np.concatenate((image, eroded_image, dilated_image), axis=2)
            if dataset_name != 'M':
                image = cv2.resize(image, target_siz)

            image = image / 255

            image = np.transpose(image, (2, 0, 1))
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

            # image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
            # val_outputs1 = sliding_window_inference(image_tensor, roi_size, sw_batch_size, model)
            # val_outputs1 = [post_trans(i) for i in decollate_batch(val_outputs1)]

            # Get prediction
            if weights_list != None:
                prediction = validate_with_augmentations_and_ensembling(model, image_tensor, weights_list)
                prediction = torch.unsqueeze(prediction, 0)
            elif augment_all:
                prediction = validate_with_augmentations(model, image_tensor)
                # prediction = prediction[0]
            else:
                prediction = model(image_tensor)
                prediction = F.softmax(prediction, dim=1)
                if model1 is not None:
                    prediction1 = model1(image_tensor)
                    prediction1 = F.softmax(prediction1, dim=1)

                    prediction = 0.5*prediction + 0.5*prediction1
            if nuclei == 2:
                # prediction = keep_only_highest_prob(prediction = prediction)
                prediction = torch.argmax(prediction[0], dim=0).squeeze(0).cpu().numpy()
            else:
                prediction = torch.argmax(prediction[0], dim=0).squeeze(0).cpu().numpy()

                # Post-process prediction (e.g., apply softmax or argmax)
            metas_sum = 0
            metas_class = np.empty((0,5))
            primary_sum = 0
            primary_class = np.empty((0,5))
            a_con = np.zeros((5))
            for i in range(6, 11):
                metas_sum = metas_sum + np.sum(prediction == i)
                a_con[i-6] = np.sum(prediction == i)
            metas_class = np.concatenate((metas_class,a_con[np.newaxis,:]), axis=0)

            for i in range(1, 6):
                primary_sum = primary_sum + np.sum(prediction == i)
                a_con[i-1] = np.sum(prediction == i)

            primary_class = np.concatenate((primary_class,a_con[np.newaxis,:]), axis=0)
            if primary_sum > metas_sum:
                frame_type.append([file_name ,'primary', primary_sum, metas_sum])
            else:
                frame_type.append([file_name ,'metas',primary_sum, metas_sum])
            if not nuclei:
                prediction[prediction>5] = prediction[prediction>5] - 5

            colormap = {
                0: [0, 0, 0],  # Black
                1: [255, 0, 0],  # Red
                2: [0, 255, 0],  # Green
                3: [0, 0, 255],  # Blue
                4: [255, 255, 0],  # Yellow
                5: [255, 0, 255],  # Magenta
                6: [0, 125, 125],  # Black
                7: [125, 0, 125],  # Red
                8: [125, 255, 125],  # Green
                9: [125, 255, 255],  # Blue
                10: [255, 255, 255],  # Yellow
            }
            # Create an RGB image by mapping class values to colormap
            rgb_image = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
            # Save the prediction as a TIF file
            with tifffile.TiffWriter(output_path) as tif:
                tif.write(prediction.astype(np.uint8), resolution=(300, 300))

            if save_jpg:
                for class_value, color in colormap.items():
                    rgb_image[prediction == class_value] = color
                cv2.imwrite(output_path[:-4] + '.jpg', rgb_image)

            if file_path is not None:
                return prediction,primary_class,metas_class

            # print(f"Processed and saved: {file_name}")
    # ground_truth_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_ground_truth2"
    prediction_folder = output_folder #"/home/ntorbati/PycharmProjects/pythonProject/validation_prediction2"
    image_shape = (1024, 1024)  # Adjust based on your images' resolution
    with open(output_folder+'/tissue_type.txt', 'w') as file:
        for sublist in frame_type:
            file.write(" ".join(map(str, sublist)) + "\n")
    dice_scores, mean_puma_dice, mean_dice_classes = compute_dice_scores(ground_truth_folder, prediction_folder,
                                                                         image_shape,nuclei)
    np.save(output_folder+'/tissue_type_metas.npy', metas_class)
    np.save(output_folder+'/tissue_type_primary.npy', primary_class)
    # print("Overall Dice Scores:", mean_puma_dice)
    # print("Overall Mean Dice Scores:", mean_dice_classes)

    # for file, scores in dice_scores.items():
    #     print(f"{file}: {scores}")


    metrics = calculate_micro_dice_score_with_masks_eval(ground_truth_folder, prediction_folder,
                                                                         image_shape, eps=0.00001,nuclei = nuclei)


    return mean_puma_dice, metrics


def circular_augmentation(train_images, masks, target_class, r1, r2, d1):
    """
    Apply circular augmentation to the specified class in the segmentation mask.

    Parameters:
        train_images (torch.Tensor): Input tensor of training images, size (B, C, H, W).
        masks (torch.Tensor): Input tensor of mask images, size (B, H, W) (torch.long).
        target_class (int): The class on which to apply the augmentation.
        r1 (int): Minimum radius of circles.
        r2 (int): Maximum radius of circles.
        d1 (float): Density of circles (fraction of target class area to be covered).

    Returns:
        tuple: Augmented training images and masks.
    """
    # Get device from input tensors
    device = train_images.device

    # Get dimensions
    B, C, H, W = train_images.shape
    augmented_images = train_images.clone()
    augmented_masks = masks.clone()

    for b in range(B):
        # Extract the target class region from the mask
        target_region = (masks[b] == target_class)
        target_area = target_region.sum().item()

        if target_area == 0:
            continue  # Skip if the target class is not present

        # Calculate maximum allowable circle area
        max_circle_area = target_area * d1
        current_area = 0

        circles = []  # Track placed circles as (x, y, r)

        while current_area < max_circle_area:
            # Random radius
            r = np.random.randint(r1, r2 + 1)

            # Generate a random center within the valid target region
            valid_y, valid_x = torch.where(target_region)
            if len(valid_y) == 0:
                break  # Exit if no valid points are left
            idx = np.random.randint(len(valid_y))
            x, y = valid_x[idx].item(), valid_y[idx].item()

            # Check for overlap
            overlap = False
            for cx, cy, cr in circles:
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist < r + cr:
                    overlap = True
                    break
            if not overlap:
                # Generate a circular area
                yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                circle_area = ((xx - x) ** 2 + (yy - y) ** 2) <= r ** 2

                # Ensure the circle is within the target region
                circle_mask = circle_area & target_region

                new_area = circle_mask.sum().item()
                # if current_area + new_area <= max_circle_area:
                    # Apply the augmentation: set the mask to 0 and the image to 0 in the circle
                augmented_masks[b][circle_mask] = 0
                augmented_images[b][:, circle_mask] = 1

                current_area += new_area
                circles.append((x, y, r))
    return augmented_images, augmented_masks
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logits, targets):
        ce_loss = torch.nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
def upsample_tissue(image_data = None,mask_data = None):
    rows, cols = np.shape(image_data)[1], np.shape(image_data)[2]  # Replace with your image dimensions
    r, c = np.indices((rows, cols))
    upper_diag = c >= r
    lower_diag = c <= r
    upper_anti_diag = r + c < rows
    lower_anti_diag = r + c >= rows - 1
    top_half = r < rows // 2
    left_half = c < cols // 2
    masks = [upper_diag, lower_diag, upper_anti_diag, lower_anti_diag,top_half, left_half]
    masks_inds = [0, 1, 0, 4, 2, 3, 3, 3, 5, 2, 4, 5, 4, 1, 5, 3, 4, 0, 1, 2, 4, 1, 5, 5,
                  5, 1, 3, 2, 0, 5, 4, 5, 0, 2, 4, 4, 5, 5, 4, 3, 5, 1, 4, 4, 4, 4, 5, 2,
                  2, 4, 2, 5, 4, 5, 0, 1, 4, 5, 4, 1, 5, 4, 5, 4, 2, 1, 5, 4, 5, 0, 2, 0,
                  1, 0, 4, 5, 4, 5, 4, 1, 5, 5, 5, 2, 4, 4, 4, 3, 2, 4, 5, 5, 5, 5, 0, 1,
                  3, 1, 3, 4, 5, 0, 1, 0, 4, 1, 2, 4, 2, 1, 1, 5, 4, 5, 0, 3, 1, 4, 0, 2,
                  0, 1, 0, 4, 5, 5, 2, 2, 0, 1, 4, 1, 4, 4, 0, 1, 5, 4, 4, 2, 0, 1, 5,
                  2, 5, 1, 1, 2, 5, 4, 2, 4, 4, 1, 3, 4, 2, 5, 5, 5, 5, 5, 4, 5, 2, 4, 5,
                  5, 1, 3, 1, 5, 5, 5, 5, 4, 2, 5, 0, 1, 1, 4, 5, 4, 4, 5, 1, 4, 1, 4, 0,
                  2, 0, 4, 5, 2, 5, 3, 4, 5, 5, 5, 3, 5, 4, 1 ]
    masks_inds[127] = 1
    masks_inds[100] = 1
    masks_inds[2] = 4
    masks_inds[8] = 4
    masks_inds[24] = 4
    masks_inds[73] = 5
    masks_inds[97] = 2
    masks_inds[105] = 2
    masks_inds[106] = 0
    masks_inds[108] = 1
    masks_inds[112] = 5
    masks_inds[113] = 2
    masks_inds[114] = 4
    masks_inds[115] = 0
    masks_inds[117] = 1
    masks_inds[118] = 2
    masks_inds[119] = 0
    masks_inds[123] = 1
    masks_inds[126] = 0
    masks_inds[133] = 0
    masks_inds[135] = 5
    masks_inds[136] = 4
    masks_inds[137] = 2
    masks_inds[138] = 3
    masks_inds[139] = 0
    masks_inds[192] = 2
    masks_inds[198] = 5







































    for j in range(mask_data.shape[0]):
        aug_num = random.choice([0,1,2,3,5,6,7,8,9,10])
        # if (j>114):# and ((aug_num>4) or ((np.sum(mask_data[j] == 5)>0))):
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(image_data[j] / 255)
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(mask_data[j])
        #     plt.show()

        msk = masks[masks_inds[j]]
        im_new = np.zeros_like(image_data[j])
        for kk in range(image_data.shape[3]):
            im_new[:,:,kk] = image_data[j,:,:,kk]*msk
            im_new[:,:,kk] = im_new[:,:,kk] + 255*(1-msk)
        msk_new = mask_data[j]*msk
        im_old = np.zeros_like(image_data[j])

        for kk in range(image_data.shape[3]):
            im_old[:,:,kk] = image_data[j,:,:,kk]*(1-msk)
            im_old[:,:,kk] = im_old[:,:,kk] + 255*msk
        msk_old = mask_data[j]*(1-msk)
        image_data[j] = im_old
        mask_data[j] = msk_old

        # if (j>114):  # and ((aug_num>4) or ((np.sum(mask_data[j] == 5)>0))):
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(image_data[j] / 255)
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(mask_data[j])
        #     plt.imshow(mask_data[j])
        #     plt.show()
        #     print(j)

        image_data = np.concatenate((image_data, im_new[np.newaxis, :]), axis=0)
        mask_data = np.concatenate((mask_data, msk_new[np.newaxis, :]), axis=0)


    return image_data,mask_data


def copy_data_tissue(validation_indices: List[int], data_path: str, data_path1: str, save_path: str, save_path1: str,
              data_type: str, images = None, masks = None,data_names = None):
    """
    Copy files from data_path to save_path based on indices and type (metastatic or primary).

    Args:
        validation_indices (List[int]): List of indices to copy.
        data_path (str): Source directory containing the files.
        save_path (str): Destination directory to save the files.
        data_type (str): Either 'metastatic' or 'primary'.
    """
    # Ensure the save_path exists
    # if masks is not None:
    #     masks[masks > 5] = masks[masks > 5] - 5

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path1, exist_ok=True)

    CLASS_MAPPING_TISSUE_R = {
        "tissue_stroma": 1,
        "tissue_blood_vessel": 2,
        "tissue_tumor": 3,
        "tissue_epidermis": 4,
        "tissue_necrosis": 5,
    }
    class_mapping = CLASS_MAPPING_TISSUE_R
    # Determine the prefix based on the type
    if data_type == 'primary':
        if os.path.exists(save_path):
            for root, dirs, files in os.walk(save_path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(save_path)
        os.makedirs(save_path, exist_ok=True)

        if os.path.exists(save_path1):
            for root, dirs, files in os.walk(save_path1, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(save_path1)
        os.makedirs(save_path1, exist_ok=True)

    prefix = f"training_set_{data_type}_roi_"

    image_shape = (1024, 1024)
    # Iterate through the indices and copy the corresponding files
    k = 0

    for index in range(len(validation_indices)):
        if data_names is not None:
            if np.sum(data_names == 'M'):
                file_name = images[validation_indices[index]]
            else:
                file_name = data_names[validation_indices[index]].replace('_NuClick.png', '_tissue.tif')
        else:
            index = validation_indices[index] + 1
            file_name = f"{prefix}{index:03}_tissue.tif"  # Format index as three digits with leading zeros
        dest_file = os.path.join(save_path, file_name)


        if data_names is not None:
            if np.sum(data_names == 'M'):
                file_name1 = masks[validation_indices[index]]
            else:
                file_name1 = data_names[validation_indices[index]].replace('_NuClick.png', '.tif')
        else:
            file_name1 = f"{prefix}{index:03}.tif"  # Format index as three digits with leading zeros
        src_file1 = os.path.join(save_path1, file_name1)


        if np.sum(data_names == 'M'):
            shutil.copy(file_name, save_path1)
            shutil.copy(file_name1, save_path)


        else:
            input_image = images[validation_indices[index]]
            with tifffile.TiffWriter(src_file1) as tif:
                tif.write(np.array(input_image,dtype=np.uint8), resolution=(300, 300))

            gt_image = masks[validation_indices[index]]
            with tifffile.TiffWriter(dest_file) as tif:
                tif.write(np.array(gt_image,dtype=np.uint8), resolution=(300, 300))

        k += 1

def validate_with_augmentations(model, image_tensor):
    """
    Perform validation with augmentations and ensemble predictions using probabilities.

    Args:
        model: The PyTorch model.
        image_tensor: The input tensor stored in GPU.

    Returns:
        Final ensembled prediction as a NumPy array.
    """
    # Define the augmentations
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512", do_resize=False, do_rescale=False)
    def augment(tensor):
        """
        Generate 7 unique augmentations of the input tensor.
        The augmentations include:
        1. Original
        2. Rotate 90°
        3. Rotate 180°
        4. Rotate 270°
        5. Horizontal flip
        6. Horizontal flip + Rotate 90°
        7. Horizontal flip + Rotate 270°

        Args:
            tensor: Input tensor of shape (B, C, H, W).

        Returns:
            List of tensors with augmentations applied.
        """
        return [
            tensor,  # Original
            torch.rot90(tensor, k=1, dims=[2, 3]),  # Rotate 90°
            torch.rot90(tensor, k=2, dims=[2, 3]),  # Rotate 180°
            torch.rot90(tensor, k=3, dims=[2, 3]),  # Rotate 270°
            torch.flip(tensor, dims=[3]),  # Horizontal flip
            torch.rot90(torch.flip(tensor, dims=[3]), k=1, dims=[2, 3]),  # Horizontal flip + Rotate 90°
            torch.rot90(torch.flip(tensor, dims=[3]), k=2, dims=[2, 3]),  # Horizontal flip + Rotate 180°
            torch.rot90(torch.flip(tensor, dims=[3]), k=3, dims=[2, 3]),  # Horizontal flip + Rotate 270°
        ]
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
        return x

    def reverse_augment(tensor, idx):
        """
        Reverse the augmentation applied at a given index.

        Args:
            tensor: Augmented tensor of shape (C, H, W).
            idx: Index of the augmentation (0-6).

        Returns:
            Tensor with the reverse transformation applied.
        """
        if idx == 0:  # Original
            return tensor
        elif idx == 1:  # Rotate 90° (reverse by rotating 270°)
            return torch.rot90(tensor, k=3, dims=[2, 3])
        elif idx == 2:  # Rotate 180° (reverse by rotating 180°)
            return torch.rot90(tensor, k=2, dims=[2, 3])
        elif idx == 3:  # Rotate 270° (reverse by rotating 90°)
            return torch.rot90(tensor, k=1, dims=[2, 3])
        elif idx == 4:  # Horizontal flip
            return torch.flip(tensor, dims=[3])
        elif idx == 5:  # Horizontal flip + Rotate 90° (reverse rotation first, then flip)
            return torch.flip(torch.rot90(tensor, k=3, dims=[2, 3]), dims=[3])
        elif idx == 6:  # Horizontal flip + Rotate 270° (reverse rotation first, then flip)
            return torch.flip(torch.rot90(tensor, k=2, dims=[2, 3]), dims=[3])
        elif idx == 7:  # Horizontal flip + Rotate 270° (reverse rotation first, then flip)
            return torch.flip(torch.rot90(tensor, k=1, dims=[2, 3]), dims=[3])
    # Apply augmentations to the input image
    augmented_inputs = augment(image_tensor)

    # Apply the model to each augmented input and store probabilities
    probabilities = []
    for augmented_input in augmented_inputs:
        with torch.no_grad():
            try:
                try:
                    model.segformer
                    device = model.device
                except:
                    model.module.segformer
                    device = model.module.device
                # Process input images
                images = processor(images=[augmented_input[0,0:3].permute(1, 2, 0).cpu().numpy()],
                                   return_tensors="pt")  # Now it's ready for SegFormer
                images = {key: value.to(device) for key, value in images.items()}

                if augmented_input.shape[1] > 3:
                    images['pixel_values'] = torch.concatenate((images['pixel_values'], augmented_input[:, 3].unsqueeze(1)),
                                                                dim=1)

                pred = model(**images)
                pred = F.interpolate(pred.logits, size=image_tensor.size()[2:],
                                           mode='bilinear', align_corners=False)
            except:
                padded_input, pad = pad_to_multiple(augmented_input, multiple=32)
                # Forward pass with the padded image
                try:
                    pred_padded = model(padded_input)
                except:
                    pred_padded = model.val(padded_input)
                # Remove the padding to get back to the original image size
                pred = remove_pad(pred_padded, pad)
                # pred = model(augmented_input)  # Forward pass
            pred = F.softmax(pred, dim=1)  # Get probabilities
            probabilities.append(pred)

    # Reverse augmentations to align probabilities
    aligned_probabilities = [reverse_augment(prob, i) for i, prob in enumerate(probabilities)]

    # Ensemble probabilities by averaging
    stacked_probs = torch.stack(aligned_probabilities, dim=0)  # Shape: [8, C, H, W]
    averaged_probs = torch.mean(stacked_probs, dim=0)  # Shape: [C, H, W]

    # averaged_probs[0][5][averaged_probs[0][5]>0.1] = 1
    # Final prediction: Argmax over the averaged probabilities
    final_prediction = torch.argmax(averaged_probs, dim=1)  # Shape: [H, W]

    # Move to CPU and convert to NumPy
    return averaged_probs


def validate_with_augmentations1(model, image_tensor):
    """
    Perform validation with augmentations and ensemble predictions.

    Args:
        model: The PyTorch model.
        image_tensor: The input tensor stored in GPU.

    Returns:
        Final ensembled prediction as a NumPy array.
    """
    # Define the augmentations
    def augment(tensor):
        return [
            tensor,  # Original
            torch.flip(tensor, dims=[2]),  # Vertical flip
            torch.flip(tensor, dims=[3]),  # Horizontal flip
            torch.flip(tensor, dims=[2, 3]),  # Vertical + Horizontal flip
            torch.rot90(tensor, k=1, dims=[2, 3]),  # Rotate 90 degrees
            torch.rot90(tensor, k=2, dims=[2, 3]),  # Rotate 180 degrees
            torch.rot90(tensor, k=3, dims=[2, 3]),  # Rotate 270 degrees
            torch.flip(torch.rot90(tensor, k=1, dims=[2, 3]), dims=[3])  # Rotate 90 + Horizontal flip
        ]

    # Apply augmentations to the input image
    augmented_inputs = augment(image_tensor)

    # Apply the model to each augmented input and store predictions
    predictions = []
    for augmented_input in augmented_inputs:
        pred = model(augmented_input)  # Forward pass
        pred = F.softmax(pred, dim=1)  # Apply softmax
        pred = torch.argmax(pred, dim=1)  # Get class predictions
        predictions.append(pred)

    # Reverse augmentations to align predictions
    def reverse_augment(tensor, idx):
        if idx == 0:  # Original
            return tensor
        elif idx == 1:  # Vertical flip
            return torch.flip(tensor, dims=[1])
        elif idx == 2:  # Horizontal flip
            return torch.flip(tensor, dims=[2])
        elif idx == 3:  # Vertical + Horizontal flip
            return torch.flip(tensor, dims=[1, 2])
        elif idx == 4:  # Rotate 90 degrees
            return torch.rot90(tensor, k=3, dims=[1, 2])
        elif idx == 5:  # Rotate 180 degrees
            return torch.rot90(tensor, k=2, dims=[1, 2])
        elif idx == 6:  # Rotate 270 degrees
            return torch.rot90(tensor, k=1, dims=[1, 2])
        elif idx == 7:  # Rotate 90 + Horizontal flip
            return torch.rot90(torch.flip(tensor, dims=[2]), k=3, dims=[1, 2])

    # Align all augmented predictions
    aligned_predictions = [reverse_augment(pred, i) for i, pred in enumerate(predictions)]

    # Ensemble predictions by majority voting
    stacked_preds = torch.stack(aligned_predictions, dim=0)
    final_prediction = torch.mode(stacked_preds, dim=0).values  # Majority vote

    # Move to CPU and convert to NumPy
    return final_prediction.cpu().numpy()

def validate_with_augmentations_and_ensembling(model, image_tensor, weights_list=None, device = ''):
    """
    Perform validation with test-time augmentations (TTA), ensembling, and debugging visualizations.

    Args:
        model: The PyTorch model.
        image_tensor: The input tensor stored on GPU.
        weights_list: List of paths to model weight files for ensembling.

    Returns:
        Final ensembled prediction as a NumPy array.
    """

    # Define the augmentations
    def augment(tensor):
        """
        Generate 7 unique augmentations of the input tensor.
        The augmentations include:
        1. Original
        2. Rotate 90°
        3. Rotate 180°
        4. Rotate 270°
        5. Horizontal flip
        6. Horizontal flip + Rotate 90°
        7. Horizontal flip + Rotate 270°

        Args:
            tensor: Input tensor of shape (B, C, H, W).

        Returns:
            List of tensors with augmentations applied.
        """
        return [
            tensor,  # Original
            torch.rot90(tensor, k=1, dims=[2, 3]),  # Rotate 90°
            torch.rot90(tensor, k=2, dims=[2, 3]),  # Rotate 180°
            torch.rot90(tensor, k=3, dims=[2, 3]),  # Rotate 270°
            torch.flip(tensor, dims=[3]),  # Horizontal flip
            torch.rot90(torch.flip(tensor, dims=[3]), k=1, dims=[2, 3]),  # Horizontal flip + Rotate 90°
            torch.rot90(torch.flip(tensor, dims=[3]), k=2, dims=[2, 3]),  # Horizontal flip + Rotate 180°
            torch.rot90(torch.flip(tensor, dims=[3]), k=3, dims=[2, 3]),  # Horizontal flip + Rotate 270°
        ]

    def reverse_augment(tensor, idx):
        """
        Reverse the augmentation applied at a given index.

        Args:
            tensor: Augmented tensor of shape (C, H, W).
            idx: Index of the augmentation (0-6).

        Returns:
            Tensor with the reverse transformation applied.
        """
        if idx == 0:  # Original
            return tensor
        elif idx == 1:  # Rotate 90° (reverse by rotating 270°)
            return torch.rot90(tensor, k=3, dims=[2, 3])
        elif idx == 2:  # Rotate 180° (reverse by rotating 180°)
            return torch.rot90(tensor, k=2, dims=[2, 3])
        elif idx == 3:  # Rotate 270° (reverse by rotating 90°)
            return torch.rot90(tensor, k=1, dims=[2, 3])
        elif idx == 4:  # Horizontal flip
            return torch.flip(tensor, dims=[3])
        elif idx == 5:  # Horizontal flip + Rotate 90° (reverse rotation first, then flip)
            return torch.flip(torch.rot90(tensor, k=3, dims=[2, 3]), dims=[3])
        elif idx == 6:  # Horizontal flip + Rotate 270° (reverse rotation first, then flip)
            return torch.flip(torch.rot90(tensor, k=2, dims=[2, 3]), dims=[3])
        elif idx == 7:  # Horizontal flip + Rotate 270° (reverse rotation first, then flip)
            return torch.flip(torch.rot90(tensor, k=1, dims=[2, 3]), dims=[3])
    # Initialize final probability predictions
    model.to(device)
    model.eval()
    final_probabilities = None  # Will hold the ensembled probabilities

    # Iterate through each model weight
    for k, weight_path in enumerate(weights_list):
        # Load model weights
        model.load_state_dict(torch.load(weight_path))

        # Apply augmentations to the input tensor
        augmented_inputs = augment(image_tensor)

        # Apply the model to each augmented input and store predictions
        augmented_probabilities = []
        for i, augmented_input in enumerate(augmented_inputs):
            augmented_input = augmented_input.to(device)
            pred = model(augmented_input)  # Forward pass
            prob = F.softmax(pred, dim=1)  # Get probabilities
            augmented_probabilities.append(prob)

            # Debugging: Visualize forward-augmented probability maps
            # plt.figure(figsize=(6, 4))
            # plt.title(f"Forward Augmentation {i}")
            # plt.imshow(prob[0].cpu().detach().numpy()[0], cmap="viridis")
            # plt.colorbar()
            # plt.show()

        # Reverse augmentations to align predictions
        aligned_probabilities = []
        for i, prob in enumerate(augmented_probabilities):
            reversed_prob = reverse_augment(prob, i)
            aligned_probabilities.append(reversed_prob)

            # Debugging: Visualize reverse-augmented probability maps
            # plt.figure(figsize=(6, 4))
            # plt.title(f"Reversed Augmentation {i}")
            # plt.imshow(reversed_prob[0].cpu().detach().numpy()[0], cmap="viridis")
            # plt.colorbar()
            # plt.show()

        # Ensure alignment of dimensions
        aligned_probabilities = torch.stack([p.squeeze(0) for p in aligned_probabilities], dim=0)

        # Average probabilities across augmentations for TTA
        tta_probability = torch.mean(aligned_probabilities, dim=0)

        # Add TTA probability to the ensemble
        if final_probabilities is None:
            final_probabilities = tta_probability
        else:
            final_probabilities += tta_probability

    # Average probabilities across all models in the ensemble
    final_probabilities /= len(weights_list)
    # final_probabilities[5][final_probabilities[5]<0.6] = 0
    # Final prediction (class-wise argmax)
    final_predictions = torch.argmax(final_probabilities, dim=0)

    # Move to CPU and convert to NumPy
    return final_probabilities



def validate_with_augmentations_and_ensembling1(model, image_tensor, weights_list=None,device = ''):
    """
    Perform validation with test-time augmentations and model ensembling.

    Args:
        model: The PyTorch model.
        image_tensor: The input tensor stored in GPU.
        weights_list: List of paths to model weight files for ensembling.

    Returns:
        Final ensembled prediction as a NumPy array.
    """

    # Define the augmentations
    def augment(tensor):
        return [
            tensor,  # Original
            torch.flip(tensor, dims=[2]),  # Vertical flip
            torch.flip(tensor, dims=[3]),  # Horizontal flip
            torch.flip(tensor, dims=[2, 3]),  # Vertical + Horizontal flip
            torch.rot90(tensor, k=1, dims=[2, 3]),  # Rotate 90 degrees
            torch.rot90(tensor, k=2, dims=[2, 3]),  # Rotate 180 degrees
            torch.rot90(tensor, k=3, dims=[2, 3]),  # Rotate 270 degrees
            torch.flip(torch.rot90(tensor, k=1, dims=[2, 3]), dims=[3])  # Rotate 90 + Horizontal flip
        ]

    # Reverse augmentations to align predictions
    def reverse_augment(tensor, idx):
        if idx == 0:  # Original
            return tensor
        elif idx == 1:  # Vertical flip
            return torch.flip(tensor, dims=[1])
        elif idx == 2:  # Horizontal flip
            return torch.flip(tensor, dims=[2])
        elif idx == 3:  # Vertical + Horizontal flip
            return torch.flip(tensor, dims=[1, 2])
        elif idx == 4:  # Rotate 90 degrees
            return torch.rot90(tensor, k=3, dims=[1, 2])
        elif idx == 5:  # Rotate 180 degrees
            return torch.rot90(tensor, k=2, dims=[1, 2])
        elif idx == 6:  # Rotate 270 degrees
            return torch.rot90(tensor, k=1, dims=[1, 2])
        elif idx == 7:  # Rotate 90 + Horizontal flip
            return torch.rot90(torch.flip(tensor, dims=[2]), k=3, dims=[1, 2])

    # Initialize final probability predictions
    model.to(device)
    model.eval()
    final_probabilities = None  # Will hold the ensembled probabilities
    preds = []
    # Iterate through each model weight
    for k, weight_path in enumerate(weights_list):
        # Load model weights
        model.load_state_dict(torch.load(weight_path))
        model.eval()

        # Apply augmentations to the input tensor
        augmented_inputs = augment(image_tensor)

        # Apply the model to each augmented input and store predictions
        augmented_predictions = []
        for augmented_input in augmented_inputs:
            augmented_input = augmented_input.to(device)
            pred = model(augmented_input)  # Forward pass
            pred = F.softmax(pred, dim=1)  # Apply softmax
            augmented_predictions.append(pred)

        # Reverse augmentations to align predictions
        aligned_predictions = [
            reverse_augment(torch.argmax(pred, dim=1), i) for i, pred in enumerate(augmented_predictions)
        ]

        # Aggregate augmented predictions
        stacked_preds = torch.stack(aligned_predictions, dim=0)
        tta_prediction = torch.mode(stacked_preds, dim=0).values  # Majority vote for TTA

        # Add TTA prediction to the ensemble
        preds.append(tta_prediction)

    stacked_preds = torch.stack(preds, dim=0)
    final_predictions = torch.mode(stacked_preds, dim=0).values  # Majority vote for TTA

    return final_predictions[0].cpu().numpy()

def compute_puma_dice_micro_dice_prediction(model = None, target_siz = None,input_path = '', device = None, weights_list = None, augment_all = True):
    # input_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_images2"
    # output_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_prediction2"
    # Read the TIF image
    image = tifffile.imread(input_path)

    # Ensure the image has 3 channels
    # Handle 4-channel images by dropping the alpha channel
    if image.shape[2] == 4:
        image = image[:, :, :3]  # Keep only the first three channels (RGB)
    elif image.shape[2] != 3:
        raise ValueError(f"Unexpected number of channels in image:")


    # image = cv2.resize(image, target_siz)

    image = image / 255

    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

    # image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    # val_outputs1 = sliding_window_inference(image_tensor, roi_size, sw_batch_size, model)
    # val_outputs1 = [post_trans(i) for i in decollate_batch(val_outputs1)]

    # Get prediction
    if weights_list != None:
        prediction = validate_with_augmentations_and_ensembling(model, image_tensor, weights_list, device=device)
    elif augment_all:
        prediction = validate_with_augmentations(model, image_tensor)
        # prediction = prediction[0]
    else:
        prediction = model(image_tensor)
        prediction = F.softmax(prediction, dim=1)
    prediction = torch.argmax(prediction[0], dim=0).squeeze(0).cpu().numpy()

    # print(f"Processed and saved: {file_name}")


    return prediction

def save_prediction_for_dice(output_folder = '', file_name = '',prediction = None, save_jpg = False):
    # Post-process prediction (e.g., apply softmax or argmax)
    output_path = os.path.join(output_folder, file_name)

    colormap = {
        0: [0, 0, 0],  # Black
        1: [255, 0, 0],  # Red
        2: [0, 255, 0],  # Green
        3: [0, 0, 255],  # Blue
        4: [255, 255, 0],  # Yellow
        5: [255, 0, 255],  # Magenta
    }

    # Create an RGB image by mapping class values to colormap
    rgb_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    # Save the prediction as a TIF file
    with tifffile.TiffWriter(output_path) as tif:
        tif.write(prediction.astype(np.uint8), resolution=(300, 300))

    if save_jpg:
        for class_value, color in colormap.items():
            rgb_image[prediction == class_value] = color
        cv2.imwrite(output_path[:-4] + '.jpg', rgb_image)


def compute_puma_dice_micro_dice_from_folder(output_folder = '', ground_truth_folder = ''):
    # ground_truth_folder = "/home/ntorbati/PycharmProjects/pythonProject/validation_ground_truth2"
    prediction_folder = output_folder #"/home/ntorbati/PycharmProjects/pythonProject/validation_prediction2"
    image_shape = (1024, 1024)  # Adjust based on your images' resolution

    dice_scores, mean_puma_dice, mean_dice_classes = compute_dice_scores(ground_truth_folder, prediction_folder,
                                                                         image_shape)
    # print("Overall Dice Scores:", mean_puma_dice)
    # print("Overall Mean Dice Scores:", mean_dice_classes)

    # for file, scores in dice_scores.items():
    #     print(f"{file}: {scores}")


    metrics = calculate_micro_dice_score_with_masks(ground_truth_folder, prediction_folder,
                                                                         image_shape, eps=0.00001)
    return mean_puma_dice, metrics

def fill_background_holes_batch(masks, max_hole_size=5000):
    """
    Fill background holes for each class in a batch of labeled masks, with a size threshold.

    Args:
        masks (numpy.ndarray): Input labeled masks of shape (N, H, W) with values [0-5].
        max_hole_size (int): Maximum size of holes to be filled.

    Returns:
        numpy.ndarray: Masks with background holes filled, same shape as input.
    """
    # Create an output array with the same shape as input
    filled_masks = masks.copy()

    # Loop through each mask in the batch
    for i in range(masks.shape[0]):
        mask = masks[i]  # Current mask (H, W)

        # Get unique class labels (excluding background, 0)
        class_labels = np.unique(mask)
        class_labels = class_labels[class_labels != 0]

        for cls in class_labels:
            # Create a binary mask for the current class
            class_binary = (mask == cls).astype(np.uint8) * 255

            # Invert the class binary to identify background holes
            inverted_mask = cv2.bitwise_not(class_binary)

            # Find connected components in the inverted mask
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_mask, connectivity=8)

            # Iterate through connected components (excluding the background, label 0)
            for label in range(1, num_labels):
                # Get the size of the current hole
                hole_size = stats[label, cv2.CC_STAT_AREA]

                # Fill the hole only if its size is less than or equal to the threshold
                if hole_size <= max_hole_size:
                    class_binary[labels == label] = 255

            # Update the original mask with the filled regions for the current class
            filled_masks[i][class_binary > 0] = cls

    return filled_masks


def modify_model_for_new_classes(model = None, n_classes = None):
    decoder_output_channels = model.segmentation_head[0].in_channels
    model.segmentation_head = smp.base.SegmentationHead(
        in_channels=decoder_output_channels,  # Decoder's final output channels
        out_channels=n_classes,  # Number of new classes
        activation=None  # Raw logits
    )
    # Initialize new segmentation head weights
    def initialize_head_weights(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    model.segmentation_head.apply(initialize_head_weights)
    return model


def random_sub_image_sampling(image_batch: torch.Tensor, mask_batch: torch.Tensor, sub_size: int):
    """
    Randomly selects sub-images from the image and mask batches.

    Args:
        image_batch (torch.Tensor): Batch of images with shape (B, C, H, W)
        mask_batch (torch.Tensor): Batch of masks with shape (B, H, W)
        sub_size (int): Size of the sub-image (assumes square sub-images)

    Returns:
        torch.Tensor: Sub-image batch of shape (B, C, sub_size, sub_size)
        torch.Tensor: Sub-mask batch of shape (B, sub_size, sub_size)
    """
    B, C, H, W = image_batch.shape
    _, H_m, W_m = mask_batch.shape

    assert H == H_m and W == W_m, "Image and mask dimensions must match."
    assert sub_size <= H and sub_size <= W, "Sub-image size must be within the image dimensions."

    # Randomly select top-left coordinates for cropping for each image independently
    top = torch.randint(0, H - sub_size + 1, (B,))
    left = torch.randint(0, W - sub_size + 1, (B,))

    # Extract sub-images and sub-masks
    sub_images = torch.stack(
        [image_batch[i, :, top[i]:top[i] + sub_size, left[i]:left[i] + sub_size] for i in range(B)])
    sub_masks = torch.stack([mask_batch[i, top[i]:top[i] + sub_size, left[i]:left[i] + sub_size] for i in range(B)])

    return sub_images, sub_masks

def add_classes_metas(mask_data_metas = None, num_classes = 6):
    for i in range(num_classes):
        mask_data_metas[mask_data_metas == 1] = 6
        mask_data_metas[mask_data_metas == 2] = 7
        mask_data_metas[mask_data_metas == 3] = 8
        mask_data_metas[mask_data_metas == 5] = 10

    return mask_data_metas


def adapt_checkpoint(checkpoint, model):
    model_dict = model.state_dict()
    new_checkpoint = {}
    for key, value in checkpoint.items():
        # print(f"Skipping {key} due to shape mismatch")
        if key in model_dict and model_dict[key].shape == value.shape:
            new_checkpoint[key] = value
        else:
            print(f"Skipping {key} due to shape mismatch")
    return new_checkpoint

def copy_data(validation_indices: List[int], data_path: str, data_path1: str, save_path: str, save_path1: str, data_type: str, masks = None, tissue = True):
    """
    Copy files from data_path to save_path based on indices and type (metastatic or primary).

    Args:
        validation_indices (List[int]): List of indices to copy.
        data_path (str): Source directory containing the files.
        save_path (str): Destination directory to save the files.
        data_type (str): Either 'metastatic' or 'primary'.
    """
    # Ensure the save_path exists
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path1, exist_ok=True)

    CLASS_MAPPING_TISSUE_R = {
        "tissue_stroma": 1,
        "tissue_blood_vessel": 2,
        "tissue_tumor": 3,
        "tissue_epidermis": 4,
        "tissue_necrosis": 5,
    }
    class_mapping = CLASS_MAPPING_TISSUE_R
    # Determine the prefix based on the type
    if data_type == 'primary':
        if os.path.exists(save_path):
            for root, dirs, files in os.walk(save_path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(save_path)
        os.makedirs(save_path, exist_ok=True)

        if os.path.exists(save_path1):
            for root, dirs, files in os.walk(save_path1, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(save_path1)
        os.makedirs(save_path1, exist_ok=True)



    prefix = f"training_set_{data_type}_roi_"

    image_shape = (1024,1024)
    # Iterate through the indices and copy the corresponding files
    for index in validation_indices:
        index+=1
        file_name = f"{prefix}{index:03}_tissue.geojson"  # Format index as three digits with leading zeros
        src_file = os.path.join(data_path, file_name)
        dest_file = os.path.join(save_path, file_name)

        file_name1 = f"{prefix}{index:03}.tif"  # Format index as three digits with leading zeros
        src_file1 = os.path.join(data_path1, file_name1)


        if os.path.exists(src_file):
            file_path = src_file
            if tissue:
                with open(file_path, 'r') as geojson_file:
                    try:
                        data = json.load(geojson_file)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid GeoJSON file: {file_name}")
                        continue

                    # Create temporary maps for the current GeoJSON file
                    current_class_map = np.zeros(image_shape, dtype=np.uint8)
                    current_instance_map = np.zeros(image_shape, dtype=np.uint32)
                    # Iterate over features in the GeoJSON
                    for i, feature in enumerate(data['features']):
                        geometry = shape(feature['geometry'])
                        class_name = feature['properties']['classification']['name']

                        # Rasterize the geometry onto the instance and class maps
                        mask = rasterize(
                            [(geometry, 1)],
                            out_shape=image_shape,
                            fill=0,
                            default_value=1,
                            dtype=np.uint8
                        )
                        current_instance_map[mask == 1] = i + 1  # Assign unique instance IDs
                        if class_name in class_mapping:
                            current_class_map[mask == 1] = class_mapping[class_name]
            else:
                current_class_map = np.array(masks[index-1],dtype=np.uint8)


            tif_file_name = f"{prefix}{index:03}_tissue.tif"
            tif_save_path = os.path.join(save_path, tif_file_name)

            min_val, max_val = current_class_map.min(), current_class_map.max()

            with tifffile.TiffWriter(tif_save_path) as tif:
                tif.write(
                    current_class_map,
                    resolution=(300, 300),  # Set resolution to 300 DPI for both X and Y
                    extratags=[
                        ('MinSampleValue', 'I', 1, int(min_val)),
                        ('MaxSampleValue', 'I', 1, int(max_val)),
                    ]
                )

                # print(f"Saved TIFF: {tif_save_path}")
            shutil.copy(src_file1, save_path1)
            # print(f"Copied: {src_file} -> {save_path1}")
        else:
            a= 0
            # print(f"File not found: {src_file}")


def save_all_csv_ocelot(pred_path = '',gt_path = ''):
    pred_paths = [f for f in os.listdir(pred_path) if f.endswith('.csv')]
    num_images = len(pred_paths)

    gt_json = {
        "type": "Multiple points",
        "num_images": num_images,
        "points": [],
        "version": {
            "major": 1,
            "minor": 0,
        }
    }

    p_json = {
        "type": "Multiple points",
        # "num_images": num_images,
        "points": [],
        "version": {
            "major": 1,
            "minor": 0,
        }
    }


    for idx, pathn in enumerate(pred_paths):
        path = os.path.join(pred_path, pathn)

        with open(path, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            x, y, c = line.split(",")
            point = {
                "name": f"image_{idx}",
                "point": [int(x), int(y), int(c)],
                "probability": 1.0,  # dummy value, since it is a GT, not a prediction
            }
            p_json["points"].append(point)

        g_path = os.path.join(gt_path, pathn)
        with open(g_path, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            x, y, c = line.split(",")
            point = {
                "name": f"image_{idx}",
                "point": [int(x), int(y), int(c)],
                "probability": 1.0,  # dummy value, since it is a GT, not a prediction
            }
            gt_json["points"].append(point)



    with open(pred_path + '/preds.json', "w") as g:
        json.dump(p_json, g)
    with open(pred_path + '/gts.json', "w") as g:
        json.dump(gt_json, g)




def ocelot_f1(gt_folder,pred_folder,inst_pth,nuclei = 3):
    images_nuclei = [f for f in os.listdir(pred_folder) if f.endswith('.tif')]

    for image in images_nuclei:
        im_pth = os.path.join(pred_folder,image)
        ins_pth = os.path.join(inst_pth,image)
        ins_pth = ins_pth.replace('.tif','.npy')
        # ins_pth = ins_pth.replace('.tif','_tissue.tif')
        # org_im = cv2.imread(os.path.join('/home/ntorbati/STORAGE/ocelot2023_v1.0.1/processed_csvs/val/cell/images',image.replace('tif','jpg')))
        # plt.imshow(org_im)
        # plt.show()

        save_f1_json(im_pth, ins_pth, pred_folder,image,nuclei = nuclei)
        save_all_csv_ocelot(pred_path=pred_folder, gt_path=gt_folder)
    f1 = ocelot_f1_main(algorithm_output_path=pred_folder + '/preds.json',gt_path=pred_folder + '/gts.json')
    return f1

def erode_connected_components(image):
    eroded_image = np.zeros_like(image)
    # Find connected components
    num_labels, labels = cv2.connectedComponents(np.array(image,dtype=np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
    3, 3))  # Circular erosion kernel of 1-pixel width  # Erosion kernel of 1-pixel width

    for label in range(1, num_labels):  # Skip background (label 0)
        mask = (labels == label).astype(np.uint8)
        eroded_component = cv2.erode(mask, kernel, iterations=2)
        eroded_image[eroded_component > 0] = label
    return eroded_image



def report_results_f1(results, counter):
    # Compute average metrics (macro F1-score for nuclei and average DICE for tissue)
    f1_scores_per_class = {}
    dice_scores_per_class = {}
    TP_per_class = {}
    FP_per_class = {}
    FN_per_class = {}
    for result in results:
        nuclei_metrics = result
        for class_name, class_metrics in nuclei_metrics.items():
            if class_name not in ["micro", "macro"]:  # skip "micro" and "macro"
                if class_name not in f1_scores_per_class:  # initialize if not in dict
                    f1_scores_per_class[class_name] = 0
                    TP_per_class[class_name] = 0
                    FP_per_class[class_name] = 0
                    FN_per_class[class_name] = 0
                    dice_scores_per_class[class_name + "len"] = 0
                # if class_name == np.str_('nuclei_epithelium'):
                    # print(class_metrics['f1_score'])
                f1_scores_per_class[class_name] += class_metrics['f1_score']
                TP_per_class[class_name] += class_metrics['TP']
                FP_per_class[class_name] += class_metrics['FP']
                FN_per_class[class_name] += class_metrics['FN']
                dice_scores_per_class[class_name + "len"] += 1

    # Compute the average F1-score for each nuclei class
    f1_scores_per_class1 = f1_scores_per_class.copy()
    f2_scores_per_class1 = f1_scores_per_class.copy()
    for class_name in f1_scores_per_class:
        f1_scores_per_class1[class_name] /= (counter)#dice_scores_per_class[class_name + "len"]
        precision = TP_per_class[class_name] / (TP_per_class[class_name] + FP_per_class[class_name]+ 1e-8)
        recall = TP_per_class[class_name] / (TP_per_class[class_name] + FN_per_class[class_name]+ 1e-8)
        f2_scores_per_class1[class_name] = 2 * (precision * recall) / (precision + recall+ 1e-8)
        f1_scores_per_class[class_name] /= dice_scores_per_class[class_name + "len"]



    print('f1_nuclei' , f1_scores_per_class)
    macro_f1 = np.mean(list(f1_scores_per_class.values()))
    print('macro_f1' , macro_f1)

    print('f1_nuclei' , f1_scores_per_class1)
    macro_f2 = np.mean(list(f1_scores_per_class1.values()))
    print('macro_f1' , macro_f2)

    macro_f3 = np.mean(list(f2_scores_per_class1.values()))
    print('macro_f1' , f2_scores_per_class1)
    print('macro_f1' , macro_f3)


   # Compute overall macro F1-score by averaging the per-class F1-scores

    return f1_scores_per_class, macro_f3


import numpy as np
import cv2


def merge_instance_maps_vectorized(map1 = None, map2 = None):
    """
    Merge two instance maps based on overlapping connected components using array operations.

    For each connected component in map1:
      - If it overlaps with exactly one connected component in map2,
        that component in map2 is removed (set to 0).
      - Otherwise (0 or more than one overlap), the component in map1 is removed.

    Parameters:
        map1, map2 (np.ndarray): Binary maps (or any maps where connectedComponents makes sense)

    Returns:
        new_labels1, new_labels2 (np.ndarray): The modified label maps for map1 and map2.
    """
    # Get connected components for both maps
    map1 = np.array(map1, dtype=np.uint8)
    map2 = np.array(map2, dtype=np.uint8)
    num_labels1, labels1 = cv2.connectedComponents(map1.astype(np.uint8))
    num_labels2, labels2 = cv2.connectedComponents(map2.astype(np.uint8))

    # We'll build an overlap matrix of shape (num_labels1, num_labels2)
    # For every pixel, combine the label from map1 and map2 into one index:
    flat1 = labels1.flatten()
    flat2 = labels2.flatten()
    M = num_labels2  # factor to combine the two labels
    combined = flat1 * M + flat2  # unique index for each (label1, label2) pair
    # Count occurrences of each pair
    overlap_counts = np.bincount(combined, minlength=num_labels1 * num_labels2)
    overlap_matrix = overlap_counts.reshape(num_labels1, num_labels2)

    # For each label in map1 (ignore background label 0), determine the overlapping labels in map2.
    # We ignore map2's background (column 0).
    remove_from_map1 = []  # Labels in map1 to remove.
    remove_from_map2 = []  # Labels in map2 to remove.

    # Loop only over labels in map1 (starting from 1)
    for label1 in range(1, num_labels1):
        # Get indices of map2 labels (excluding background) that overlap with label1
        overlapping = np.where(overlap_matrix[label1, 1:] > 0)[0] + 1  # adjust index (+1)
        if len(overlapping) == 1:
            # If exactly one overlap, remove that component from map2.
            remove_from_map2.append(overlapping[0])
        else:
            # Otherwise, remove the component from map1.
            remove_from_map1.append(label1)

    # Create new label maps, setting removed components to 0
    new_labels1 = labels1.copy()
    new_labels2 = labels2.copy()

    if remove_from_map1:
        # Vectorized removal: set pixels in labels1 that are in remove_from_map1 to 0
        mask = np.isin(new_labels1, remove_from_map1)
        new_labels1[mask] = 0

    if remove_from_map2:
        mask = np.isin(new_labels2, remove_from_map2)
        new_labels2[mask] = 0

    return new_labels1, new_labels2



def merge_instance_maps1(map1=None, map2=None):
    # Find connected components in both maps
    map1 = np.array(map1, dtype=np.uint8)
    map2 = np.array(map2, dtype=np.uint8)
    num_labels1, labels1 = cv2.connectedComponents(map1)
    num_labels2, labels2 = cv2.connectedComponents(map2)

    # Precompute masks for all labels in map2
    label_masks2 = {label: (labels2 == label) for label in range(1, num_labels2)}

    for label1 in range(1, num_labels1):
        mask1 = (labels1 == label1)
        overlapping_labels = set()

        # Efficiently find overlaps by checking intersection with all precomputed masks
        for label2, mask2 in label_masks2.items():
            overlap = np.any(mask1 & mask2)
            if overlap:
                overlapping_labels.add(label2)

        if len(overlapping_labels) == 1:
            # Keep the component in map1 and remove from map2
            labels2[labels2 == list(overlapping_labels)[0]] = 0
        else:
            # Remove the component from map1
            labels1[labels1 == label1] = 0

    return labels1, labels2

def recover_inst(img = None,recenter = 0.9,output_size = (1024,1024)):
    if len(img.shape) == 2:
        h, w = img.shape  # Grayscale image
    else:
        h, w, _ = img.shape  # Color image

    # Calculate original dimensions using the recenter factor
    orig_h, orig_w = int(h * recenter), int(w * recenter)

    # Calculate the coordinates to extract the original image from the padded one
    top = (h - orig_h) // 2
    left = (w - orig_w) // 2

    # Crop the image to the original size
    cropped_img = img[top:top + orig_h, left:left + orig_w]

    # Resize the cropped image to the specified size (1024x1024 by default)
    resized_img = cv2.resize(cropped_img, output_size, interpolation=cv2.INTER_NEAREST)
    return resized_img

def create_csv_ocelot(nuclei_pth='', inst_pth='', nuclei=3):
    full_instance_map1 = np.array(np.load(inst_pth),dtype = np.uint8)[:,:,0]
    HoverNextClassMap = np.array(np.load(inst_pth),dtype = np.uint8)[:,:,1]
    # full_instance_map1 = recover_inst(img=full_instance_map1, recenter=0.9, output_size=(1024, 1024))

    # full_instance_map1 = Image.open(inst_pth.replace('.png','.tif')).convert("L")

    class_map = cv2.cvtColor(cv2.imread(nuclei_pth), cv2.COLOR_BGR2GRAY)
    class_map1 = np.copy(class_map)
    # class_map1 = HoverNextClassMap
    # plt.imshow(class_map1)
    # plt.show()
    # full_instance_map1 = merge_instance_maps(full_instance_map1, class_map)

    # full_instance_map1 = dilate_erode(image=full_instance_map1, disk_radius=5, itersations=1, dilate=False)
    # full_instance_map1 = cv2.connectedComponents(full_instance_map1)[1]
    # full_instance_map1 = erode_connected_components(full_instance_map1)

    # labels1, labels2 = merge_instance_maps_vectorized(map1 = full_instance_map1, map2 = full_instance_map)

    # full_instance_map = np.array(labels1>0,dtype = np.uint8)+ np.array(labels2>0,dtype = np.uint8)
    class_map = merge_instance_maps(full_instance_map1, class_map1)
    class_names = {
    1: 'nuclei_other',
    2: 'nuclei_tumor',
}

    output_file = nuclei_pth.replace('.tif','.csv')

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        for class_id in range(1, len(class_names) + 1):
            # Create binary mask for current class
            class_mask = (class_map == class_id).astype(np.uint8)

            # Find outlines of binary mask
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask)

            for i in range(1, num_labels):  # Skip background (label 0)
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                # contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #
            # for contour in contours:
            #     # Convert contour points to the desired format (x, y, class_id)
            #     M = cv2.moments(contour)
            #     cx = int(M["m10"] / M["m00"])  # X coordinate of centroid
            #     cy = int(M["m01"] / M["m00"])
                path_points = [int(cx), int(cy), class_id]

                # Write each point to the CSV file
                writer.writerow(path_points)

    # print(f"Patch points saved to {output_file}")


def save_f1_json(nuclei_pth, inst_pth,out_folder,name,nuclei = 3, pred3from10 = False):
    if nuclei == 2:
        create_csv_ocelot(nuclei_pth=nuclei_pth, inst_pth=inst_pth, nuclei=nuclei)

    else:
        output_json = create_polygon_json_ours(nuclei_pth=nuclei_pth, inst_pth=inst_pth, nuclei=nuclei, pred3from10=pred3from10)
        # output_json = create_polygon_json(pinst_out, pcls_out, params)
        # add version (required by GC)
        output_json["version"] = {
            "major": 1,
            "minor": 0
        }

        # save JSON file
        # json_filename = os.path.join('/output/melanoma-3-class-nuclei-segmentation.json')
        if pred3from10:
            name = name.replace('.tif','_3.json')
        else:
            name = name.replace('.tif','.json')

        json_filename = os.path.join(out_folder, name)
        with open(json_filename, "w") as fp:
            json.dump(output_json, fp, indent=2)
        # print(f"JSON file saved to {json_filename}")
        # Debug: Verify file was saved
        if os.path.exists(json_filename):
            a = 0
            # print(f"Successfully saved: {json_filename}")
        else:
            print(f"Failed to save: {json_filename}")


def convert2threecell(class_map):
    """
    Convert a 10-class nuclei segmentation map to a 3-class segmentation map.

    Args:
        class_map (numpy array): Input segmentation map with values 0-10.

    Returns:
        numpy array: Converted segmentation map with values 0-3.
    """
    # Initialize new class map with the same shape
    new_class_map = np.zeros_like(class_map)

    # Mapping from 10-class to 3-class system
    new_class_map[np.isin(class_map, [10])] = 1  # Lymphocyte -> cell_lymphocyte
    new_class_map[np.isin(class_map, [4])] = 2  # Tumor -> cell_tumor
    new_class_map[np.isin(class_map, [1, 2, 3, 5, 6, 7, 8, 9])] = 3  # Others -> cell_other

    return new_class_map

def create_polygon_json_ours(nuclei_pth, inst_pth, nuclei = 3, pred3from10 = False):
    """
    Converts the instance map and class map into a JSON structure for polygon output.

    Parameters
    ----------
    pinst_out: zarr array
        In-memory instance segmentation results.
    pcls_out: dict
        Class map containing instance-to-class mapping information.
    params: dict
        Parameter store, defined in initial main.

    Returns
    ----------
    output_json: dict
        JSON structure containing polygon data.
    """
    # inst_path = str(f"{nuc_path}").replace(".tif", "_inst.npy")
    # np.save(inst_path, full_instance_map)
    full_instance_map = np.load(inst_pth)
    class_map = cv2.cvtColor(cv2.imread(nuclei_pth), cv2.COLOR_BGR2GRAY)

    class_map = merge_instance_maps(full_instance_map, class_map)
    if pred3from10:
        class_map = convert2threecell(class_map,)
        nuclei = 3
    # Define colors and class names for the different classes
    colors = {
        1: (0, 255, 0),  # lymphocytes
        2: (255, 0, 0),  # tumor
        3: (0, 0, 255)  # other
    }

    if nuclei == 10:
        class_names = {
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
        class_names = {
        1: 'nuclei_lymphocyte',
        2: 'nuclei_tumor',
        3: 'nuclei_other'
    }



    output_json = {
        "type": "Multiple polygons",
        "polygons": []
    }

    for class_id in range(1, len(class_names)+1):
        # create binary mask for current class
        class_mask = (class_map == class_id).astype(np.uint8)

        # find outlines of binary mask
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # convert contour points to the desired format (x, y, z)
            # note: since we are dealing with 2D-polygons, we choose to set Z to 0.5
            path_points = [[float(pt[0][0]), float(pt[0][1]), 0.5] for pt in contour]

            # create a polygon entry
            polygon_entry = {
                "name": class_names[class_id],
                "seed_point": path_points[0],  # using first point as the seed point
                "path_points": path_points,
                "sub_type": "",  # empty string for subtype
                "groups": [],  # empty array for groups
                "probability": 1  # confidence score
            }
            output_json["polygons"].append(polygon_entry)

    return output_json


def merge_instance_maps(full_instance_map, class_map):
    # Label connected components in the instance map
    if len(full_instance_map.shape)>3:
        full_instance_map = full_instance_map.squeeze(0).permute(1,2,0).cpu().numpy()
        class_map = class_map.squeeze(0).permute(1,2,0).cpu().numpy()
    labeled_instances, num_features = label(full_instance_map)

    # Create an output map initialized with zeros (background)
    output_map = np.zeros_like(full_instance_map, dtype=np.uint8)

    for instance_id in range(1, num_features + 1):
        # Get the mask of the current instance
        instance_mask = labeled_instances == instance_id

        # Get the corresponding class values from the class_map
        class_values = class_map[instance_mask]

        # Perform majority voting to determine the most common class
        if np.sum(class_values > 0):
            class_values = class_values[class_values > 0]

        # if np.sum(class_values==3):
        #     class_values = class_values[class_values>0]
        #     if np.sum(class_values<3):
        #         if np.sum(class_values==2):
        #             class_values = class_values[class_values ==2]

        class_counts = Counter(class_values)
        most_common_class, _ = class_counts.most_common(1)[0]

        # If the majority class is 0 (background), assign class 3
        # if most_common_class == 0:
        #     most_common_class = 3

        # Assign the determined class to the output map
        output_map[instance_mask] = most_common_class

    return output_map


def tile_or_pad_image_and_mask(image, mask, target_size=(1024, 1024), stride=1024):
    """
    If image == target_size → return as-is
    If larger → tile
    If smaller → pad
    """
    C, H, W = image.shape
    target_h, target_w = target_size
    patches = []

    if H == target_h and W == target_w:
        # Already correct size — just return it
        patches.append((image, mask))

    elif H < target_h:
        # Padding
        pad_h = max(0, target_h - H)
        # pad_w = max(0, target_w - W)
        image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        mask_padded = F.pad(mask, (0, pad_w, 0, pad_h), mode='constant', value=0)
        patches.append((image_padded, mask_padded))
    elif W < target_w:
        pad_h = max(0, target_h - H)
        pad_w = max(0, target_w - W)
        image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        mask_padded = F.pad(mask, (0, pad_w, 0, pad_h), mode='constant', value=0)
        patches.append((image_padded, mask_padded))

    else:
        # Tiling
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y1 = y
                x1 = x
                y2 = min(y + target_h, H)
                x2 = min(x + target_w, W)

                img_patch = image[:, y1:y2, x1:x2]
                mask_patch = mask[y1:y2, x1:x2]

                pad_h = target_h - img_patch.shape[1]
                pad_w = target_w - img_patch.shape[2]

                if pad_h > 0 or pad_w > 0:
                    img_patch = F.pad(img_patch, (0, pad_w, 0, pad_h), mode='constant', value=0)
                    mask_patch = F.pad(mask_patch, (0, pad_w, 0, pad_h), mode='constant', value=0)

                patches.append((img_patch, mask_patch))

    return patches

def collate_tile_patches(batch):
    """
    Flattens the list of lists of (image, mask) tuples from each sample.
    """
    all_images = []
    all_masks = []
    for patches in batch:  # batch is a list of lists of (image, mask)
        for image, mask in patches:
            all_images.append(image)
            all_masks.append(mask)

    images = torch.stack(all_images)
    masks = torch.stack(all_masks)
    return images, masks


def Panoptic_quality1(ground_truth_image, predicted_image, iters=1):

    # Convert input images to uint8 arrays, then compute connected components.
    gt_img = np.array(ground_truth_image, dtype=np.uint8)
    _, gt_cc = cv2.connectedComponents(gt_img)
    ground_truth_image = gt_cc

    pred_img = np.array(predicted_image, dtype=np.uint8)
    pred_out = cv2.connectedComponentsWithStats(pred_img)
    predicted_image = pred_out[1]      # Label image
    stats = pred_out[2]                # Statistics: each row corresponds to a label; last column is area
    num_pred = pred_out[0]

    # Choose threshold based on the iters parameter.
    th = 400 if iters == 2 else 50

    # Vectorized filtering: remove predicted instances with area below threshold.
    # The first row in stats corresponds to background (label 0) so we only check labels 1..end.
    # small_labels = np.where(stats[1:, -1] < th)[0] + 1  # shift indices, because background is label 0
    # if small_labels.size:
    #     mask = np.isin(predicted_image, small_labels)
    #     predicted_image[mask] = 0

    # Dictionary for matched instances (ground truth label -> (predicted label, IOU)).
    matched_instances = {}

    # Get unique ground truth labels
    gt_labels = np.unique(ground_truth_image)
    for i in gt_labels:
        if i == 0:
            continue
        gt_mask = (ground_truth_image == i)
        # Look only at predicted labels overlapping this instance.
        candidate_labels = np.unique(predicted_image[gt_mask])
        for j in candidate_labels:
            if j == 0:
                continue
            pred_mask = (predicted_image == j)
            # Apply the dilate_erode process (convert mask to uint8, then threshold after processing)
            if iters > 1:
                pred_mask_processed = dilate_erode(pred_mask.astype(np.uint8), itersations=iters) > 0
            else:
                pred_mask_processed = pred_mask

            # Compute intersection and union for IOU.
            intersection = np.sum(gt_mask & pred_mask_processed)
            union = np.sum(gt_mask | pred_mask_processed)
            IOU = intersection / union if union > 0 else 0.0

            if IOU > 0.5:
                # Save the match (if several candidates meet the condition, the last one wins as in original)
                matched_instances[i] = (j, IOU)

    TP = 0
    FN = 0
    sum_IOU = 0.0
    # Use a set of predicted labels to track unmatched ones.
    pred_set = set(np.unique(predicted_image))
    pred_set.discard(0)

    # Loop over all ground truth labels to count true positives and false negatives.
    for i in gt_labels:
        if i == 0:
            continue
        if i in matched_instances:
            pred_label, iou = matched_instances[i]
            pred_set.discard(pred_label)
            TP += 1
            sum_IOU += iou
        else:
            FN += 1

    FP = len(pred_set)
    denominator = TP + 0.5 * FP + 0.5 * FN
    PQ = sum_IOU / denominator if denominator != 0 else 0

    return PQ

# Compute Panoptic quality metric for each image
def Panoptic_quality(ground_truth_image, predicted_image,iters = 1):
    TP = 0
    FP = 0
    FN = 0
    sum_IOU = 0
    matched_instances = {}  # Create a dictionary to save ground truth indices in keys and predicted matched instances as velues
    # It will also save IOU of the matched instance in [indx][1]
    ground_truth_image = cv2.connectedComponents(np.array(ground_truth_image,dtype = np.uint8))[1]

    predicted_image1 = cv2.connectedComponentsWithStats(np.array(predicted_image,dtype = np.uint8))
    predicted_image = predicted_image1[1]
    if iters == 2:
        th = 400
    else:
        th = 25
    for kj in range(1,predicted_image1[0]):
        if predicted_image1[2][kj,-1] < th:
            predicted_image[predicted_image == kj] = 0

    # Find matched instances and save it in a dictionary
    for i in np.unique(ground_truth_image):
        if i == 0:
            pass
        else:
            temp_image = np.array(ground_truth_image)
            temp_image = temp_image == i
            matched_image = temp_image * predicted_image

            for j in np.unique(matched_image):
                if j == 0:
                    pass
                else:
                    pred_temp = predicted_image == j
                    pred_temp = dilate_erode(np.array(pred_temp, dtype=np.uint8),itersations=iters)
                    pred_temp = pred_temp > 0
                    intersection = sum(sum(temp_image * pred_temp))
                    union = sum(sum(temp_image + pred_temp))
                    IOU = intersection / union
                    if IOU > 0.5:
                        matched_instances[i] = j, IOU

                        # Compute TP, FP, FN and sum of IOU of the matched instances to compute Panoptic Quality

    pred_indx_list = np.unique(predicted_image)  # Find all predicted instances
    pred_indx_list = np.array(pred_indx_list[1:])  # Remove 0 from the predicted instances

    # Loop on ground truth instances
    for indx in np.unique(ground_truth_image):
        if indx == 0:
            pass
        else:
            if indx in matched_instances.keys():
                pred_indx_list = np.delete(pred_indx_list, np.argwhere(pred_indx_list == matched_instances[indx][0]))
                TP = TP + 1
                sum_IOU = sum_IOU + matched_instances[indx][1]
            else:
                FN = FN + 1
    FP = len(np.unique(pred_indx_list))
    PQ = sum_IOU / (TP + 0.5 * FP + 0.5 * FN)

    return PQ


# def Panoptic_quality(ground_truth_labels, predicted_labels):
#     TP = 0
#     FP = 0
#     FN = 0
#     sum_IOU = 0
#     matched_instances = {}  # Create a dictionary to save ground truth indices in keys and predicted matched instances as velues
#     # It will also save IOU of the matched instance in [indx][1]
#     ground_truth_image = cv2.connectedComponents(np.array(ground_truth_labels, dtype=np.uint8))[1]
#     predicted_image = cv2.connectedComponents(np.array(predicted_labels, dtype=np.uint8))[1]
#     # Find matched instances and save it in a dictionary
#     for i in np.unique(ground_truth_image):
#         if i == 0:
#             pass
#         else:
#             temp_image = np.array(ground_truth_image)
#             temp_image = temp_image == i
#             matched_image = temp_image * predicted_image
#
#             for j in np.unique(matched_image):
#                 if j == 0:
#                     pass
#                 else:
#                     pred_temp = predicted_image == j
#                     intersection = sum(sum(temp_image * pred_temp))
#                     union = sum(sum(temp_image + pred_temp))
#                     IOU = intersection / union
#                     if IOU > 0.5:
#                         matched_instances[i] = j, IOU
#
#                         # Compute TP, FP, FN and sum of IOU of the matched instances to compute Panoptic Quality
#
#     pred_indx_list = np.unique(predicted_image)  # Find all predicted instances
#     pred_indx_list = np.array(pred_indx_list[1:])  # Remove 0 from the predicted instances
#
#     # Loop on ground truth instances
#     for indx in np.unique(ground_truth_image):
#         if indx == 0:
#             pass
#         else:
#             if indx in matched_instances.keys():
#                 pred_indx_list = np.delete(pred_indx_list, np.argwhere(pred_indx_list == matched_instances[indx][0]))
#                 TP = TP + 1
#                 sum_IOU = sum_IOU + matched_instances[indx][1]
#             else:
#                 FN = FN + 1
#     FP = len(np.unique(pred_indx_list))
#     PQ = sum_IOU / (TP + 0.5 * FP + 0.5 * FN)
#
#     return PQ
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




def PQ_Mean(gt_folder = '', pred_folder = ''):
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(".tif")])
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(".tif")])

    # Ensure both folders have the same number of files
    # if len(gt_files) != len(pred_files):
    #     raise ValueError("Mismatch in the number of files between ground truth and predictions.")
    mean_dice = 0
    counter=0

    class_map = {1:'Epithelial',
                   2:'Lymphocyte',
                   3:'Neutrophil',
                 4: 'Macrophage',
                 5:'Ambiguous'
                   }
    all_pq1 = []
    all_pq2 = []
    meanPQ = []
    all_num = []
    inst_paths = '/home/ntorbati/PycharmProjects/hover_next_train/monusac_0/preds/inst/'
    inst_folders = os.listdir(inst_paths)
    im_names = np.load('/home/ntorbati/PycharmProjects/hover_next_train/data/data_monusac/fold_0/valid_name.npy')
    # for pths in inst_folders:
    for gt_file in gt_files:
        all_pq = {}
        gt_path = os.path.join(gt_folder, gt_file)
        pred_path = os.path.join(pred_folder, gt_file)
        # print('inference on' + gt_file)
        tif1 = np.array(Image.open(gt_path))  # '.resize(image_shape, Image.NEAREST))
        tif2 = np.array(Image.open(pred_path))  # .resize(image_shape, Image.NEAREST))


        inst =  cv2.cvtColor(cv2.imread(gt_path.replace('masks','IIAI')), cv2.COLOR_BGR2GRAY)
        inst[inst == 84] = 0
        insts = [76, 226, 29, 150]

        for iyn in range(len(insts)):
            inst[inst == insts[iyn]] = iyn + 1

        # inst = cv2.resize(inst, (tif1.shape[1],tif1.shape[0]), interpolation = cv2.INTER_NEAREST)
        # inst_ind = np.where('/home/ntorbati/STORAGE/MoNuSAC/Test_GT/images/' + gt_file == im_names)
        #
        # inst = np.load(os.path.join(os.path.join(inst_paths, pths), 'val' + str(inst_ind[0][0]) + '.npy')).astype(np.uint8)

        # tif2[tif1 == 5] = 0
        # if np.sum(inst) == 0:
        #     inst = tif2
        # inst[tif1 == 5] = 0
        ground_truth_path = '/home/ntorbati/STORAGE/MoNuSAC/all_masks/all_ims/'

        if tif1.shape != tif2.shape:
            inst = cv2.resize(inst, (tif1.shape[1], tif1.shape[0]), interpolation=cv2.INTER_NEAREST)
            im1 = tile_or_pad1(torch.tensor(np.zeros((tif1.shape[0], tif1.shape[1], 3))).permute(2, 0, 1),
                               torch.tensor(tif1), target_size=(1024, 1024),
                               pad_val=0, only_pad=True)
            tif1 = im1[0][1].cpu().numpy().astype(np.uint8)

        im1 = tile_or_pad1(torch.tensor(np.zeros((inst.shape[0], inst.shape[1], 3))).permute(2, 0, 1),
                           torch.tensor(inst), target_size=(1024, 1024),
                           pad_val=0, only_pad=True)
        inst = im1[0][1].cpu().numpy().astype(np.uint8)

        try:
            # inst = cv2.resize(inst, (tif1.shape[1], tif1.shape[0]), interpolation=cv2.INTER_NEAREST)
            tif2 = merge_instance_maps(inst, tif2)
            # tif2 = inst
        except:
            print('wtf')

        # If the ground truth (tif1) has 4 channels, use the first channel
        if tif1.ndim == 3 and tif1.shape[-1] == 4:  # Check if it's 4 channels
            tif1 = tif1[:, :, 0]  # Use the first channel (or modify as needed for your use case)

        # If the predictions (tif2) have multiple channels, use the first channel
        if tif2.ndim == 3 and tif2.shape[-1] > 1:
            tif2 = tif2[:, :, 0]  # Use the first channel (or modify as needed
        for category in np.unique(tif1):
            # Generate binary masks for each class
            if category == 0 or category == 5:
                continue
            # mask1 = np.where(tif1 == category, 1, 0)
            mask2 = np.where(tif2 == category, 1, 0)
            if category == 3:
                iters = 2
            else:
                iters = 1
            pth = ground_truth_path + gt_file.replace('.tif','') + '/' + class_map[category] + '/'
            mask1_pth = os.listdir(pth)
            if len(mask1_pth) == 0:
                print('error')
            else:
                mask1 = cv2.imread(pth + mask1_pth[0],cv2.IMREAD_UNCHANGED)
                im1 = tile_or_pad1(torch.tensor(np.zeros((mask1.shape[0], mask1.shape[1], 3))).permute(2, 0, 1),
                                   torch.tensor(mask1), target_size=(1024, 1024),
                                   pad_val=0, only_pad=True)
                mask1 = im1[0][1].cpu().numpy().astype(np.uint8)

            # mask1, num_features = scipy.ndimage.measurements.label(mask1)
            mask2, num_features = scipy.ndimage.measurements.label(mask2)

            for k in range(num_features):
                if np.sum(mask2 == k) < 60:
                    mask2[mask2 == k] = 0

            predicted_mask1 = np.zeros_like(mask2)
            for labels in range(1, num_features + 1):
                msk = (mask2 == labels).astype(np.uint8)
                msk = dilate_erode(msk, 1, 1, True)
                predicted_mask1[(msk == 1) & (predicted_mask1 == 0)] = labels
            mask2 = predicted_mask1


            PQ = get_fast_pq(remap_label(mask1), remap_label(mask2))
            # PQ = Panoptic_quality1(mask1, mask2,iters=iters)
        # all_pq.append(PQ)
            all_pq[class_map[category]] = PQ[0][2]

        for category in range(1, len(class_map) + 1):
            if category == 3 or category == 4:
                wc = 10
            else:
                wc = 1
            try:
                meanPQ.append(wc * all_pq[class_map[category]])
                all_num.append(wc)

            except:
                continue
        all_pq1.append(all_pq)
            # print(np.sum(meanPQ) / np.sum(all_num))
            # print(np.mean(np.array(all_pq2)))
    # pq = {}
    # for category in range(1, len(class_map)):
    #     val = 0
    #     count = 0
    #     for j in range(len(all_pq1)):
    #         try:
    #             val += all_pq1[j][class_map[category]]
    #             count = count + 1
    #         except:
    #             continue
    #     if count == 0:
    #         pq[class_map[category]] = 0
    #     else:
    #         pq[class_map[category]] = val/count
    # print(pq)
    pq = []
    counts = []
    for j in range(len(all_pq1)):
        count = 0
        val = 0
        for category in range(1, len(class_map)):
            if category == 3 or category == 4:
                wc = 10
            else:
                wc = 1
            try:
                val += wc*all_pq1[j][class_map[category]]
                count = count + wc*1
            except:
                continue
        pq.append(val)
        counts.append(count)
    mean1 = (np.array(pq) / np.array(counts)).mean()
    print(mean1)

    mean = np.sum(meanPQ)/np.sum(all_num)
    print(mean)

        # print(f"Dice Scores for {gt_file}: {scores}")

    return mean1

def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred

def get_fast_pq(true, pred, match_iou=0.5):
    """`match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            try:
                p_mask = pred_masks[pred_id]
            except:
                print('hi')
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]

def prepare_pq(inst_folder = '', pred_folder = '', npy = False, ours_inst = False):
    gt_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(".npy")])
    # pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(".tif")])

    for gt_file in gt_files:
        all_pq = {}

        # # this is for am and tia lab results
        if not ours_inst:
            gt_path = os.path.join(inst_folder, gt_file.replace('.npy', '.tif'))
            inst =  cv2.cvtColor(cv2.imread(gt_path.replace('masks','IIAI')), cv2.COLOR_BGR2GRAY)
            inst[inst == 84] = 0
            insts = [76, 226, 29, 150]
            gt = cv2.cvtColor(cv2.imread(gt_path.replace('IIAI','GT')), cv2.COLOR_BGR2GRAY)
            for iyn in range(len(insts)):
                inst[inst == insts[iyn]] = iyn + 1

        else:
            gt_path = os.path.join(inst_folder, gt_file.replace('.npy', '_inst.npy'))
            inst =  np.load(gt_path)



        pred_path = os.path.join(pred_folder, gt_file)
        # print('inference on' + gt_file)
        if npy:
            tif2 = np.load(pred_path.replace('.tif','.npy'))  # .resize(image_shape, Image.NEAREST))
        else:
            tif2 = np.array(Image.open(pred_path))  # .resize(image_shape, Image.NEAREST))
        # plt.subplot(1, 4, 1)
        # plt.imshow(tif2)
        tif2 = merge_instance_maps(inst, tif2)

        # plt.subplot(1, 4, 2)
        # plt.imshow(inst)
        # plt.subplot(1, 4, 3)
        # plt.imshow(gt)
        # plt.subplot(1, 4, 4)
        # plt.imshow(tif2)
        # plt.show()

        # tif2 = inst
        with tifffile.TiffWriter(pred_path.replace('.npy','.tif')) as tif:
            tif.write(tif2.astype(np.uint8), resolution=(300, 300))

def draw_instances_border(inst = None):
    radius = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    new_inst = np.array(np.zeros_like(inst[:,:,:,0]), dtype=np.double)
    for i in range(inst.shape[0]):
        im = inst[i,:,:,0]
        imn = np.zeros_like(im)
        for j in range(1,np.max(im)):
            mask = np.array((im == j), dtype=np.uint8)
            imn = imn + cv2.dilate(mask, kernel, iterations=1)
        c = np.array((imn == 2),dtype=np.uint8)
        imb = np.zeros_like(im)
        dilated1 = c
        cof = 1
        for k in range(5):
            dilated = cv2.dilate(dilated1, kernel, iterations=1)
            imb = imb + cof*c
            c = dilated - dilated1
            dilated1 = dilated
            cof = cof - 0.2
        new_inst[i,:,:] = imb

    return new_inst

def find_max_inst_size(ims = None):
    max_area = 0
    max_unique = 0
    for b in range(ims.shape[0]):
        unique, counts = np.unique(ims[b], return_counts=True)
        for label, count in zip(unique, counts):
            if label == 0:
                continue
            if count > max_area:
                max_area = count
                max_instance = (b, label)
            if max(unique) > max_unique:
                max_unique = max(unique)
    return max_area




####
def fix_mirror_padding(ann):
    """Deal with duplicated instances due to mirroring in interpolation
    during shape augmentation (scale, rotation etc.).

    """
    current_max_id = np.amax(ann)
    inst_list = list(np.unique(ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(ann == inst_id, np.uint8)
        remapped_ids = measurements.label(inst_map)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(ann)
    return ann


####
def gaussian_blur(images, random_state, parents, hooks, max_ksize=3):
    """Apply Gaussian blur to input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    ksize = random_state.randint(0, max_ksize, size=(2,))
    ksize = tuple((ksize * 2 + 1).tolist())

    ret = cv2.GaussianBlur(
        img, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REPLICATE
    )
    ret = np.reshape(ret, img.shape)
    ret = ret.astype(np.uint8)
    return [ret]


####
def median_blur(images, random_state, parents, hooks, max_ksize=3):
    """Apply median blur to input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    ksize = random_state.randint(0, max_ksize)
    ksize = ksize * 2 + 1
    ret = cv2.medianBlur(img, ksize)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_hue(images, random_state, parents, hooks, range=None):
    """Perturbe the hue of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    hue = random_state.uniform(*range)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if hsv.dtype.itemsize == 1:
        # OpenCV uses 0-179 for 8-bit images
        hsv[..., 0] = (hsv[..., 0] + hue) % 180
    else:
        # OpenCV uses 0-360 for floating point images
        hsv[..., 0] = (hsv[..., 0] + 2 * hue) % 360
    ret = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_saturation(images, random_state, parents, hooks, range=None):
    """Perturbe the saturation of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    value = 1 + random_state.uniform(*range)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret = img * value + (gray * (1 - value))[:, :, np.newaxis]
    ret = np.clip(ret, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_contrast(images, random_state, parents, hooks, range=None):
    """Perturbe the contrast of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    value = random_state.uniform(*range)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    ret = img * value + mean * (1 - value)
    ret = np.clip(img, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_brightness(images, random_state, parents, hooks, range=None):
    """Perturbe the brightness of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    value = random_state.uniform(*range)
    ret = np.clip(img + value, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]

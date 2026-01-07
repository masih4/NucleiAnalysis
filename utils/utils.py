from skimage.draw import polygon
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tifffile
from PIL import Image
import kornia.augmentation as K
import kornia.morphology as KM
import torchvision.transforms as T
import shutil
from typing import List
import json
from shapely.geometry import shape
from rasterio.features import rasterize
import torch
import xml.etree.ElementTree as ET

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



def adapt_checkpoint_dualEncoder(model,folds, dg = False):
    model_dict = model.state_dict()
    new_checkpoint = {}

    if dg:
    #encoder 1 pretrained(unet)
        fineTune_PATH_unet = '/home/ntorbati/PycharmProjects/DualNet_New/checkpoint/' + 'ours_n_model' + str(folds) + '_v0.pth'
        fineTune_PATH_segformer = '/home/ntorbati/PycharmProjects/DualNet_New/checkpoint/' + 'ours_s_model' + str(folds) + '.pth'

    else:
        fineTune_PATH_unet = '/home/ntorbati/PycharmProjects/DGAUNet/Puma' + 'ours_n' + str(folds) + str(211) + str(9) + '/checkpoint_epoch1.pth'
        fineTune_PATH_segformer = '/home/ntorbati/PycharmProjects/DGAUNet/Puma' + 'ours_s' + str(folds) + str(211) + str(9) + '/checkpoint_epoch1.pth'

    cp_unet = torch.load(fineTune_PATH_unet, weights_only=True)

    for key, value in cp_unet.items():
        # print(f"Skipping {key} due to shape mismatch")
        if key in model_dict and model_dict[key].shape == value.shape and ('unet' in key):
            new_checkpoint[key] = value
        else:
            print(f"Skipping {key} due to shape mismatch")






    #encoder 2 pretrained(segformer)
    cp_segformer = torch.load(fineTune_PATH_segformer, weights_only=True)

    # if "module." in list(cp.keys())[0]:
    #     cp = {k.replace("module.", ""): v for k, v in cp.items()}

    for key, value in cp_segformer.items():
        # print(f"Skipping {key} due to shape mismatch")
        if key in model_dict and model_dict[key].shape == value.shape and ('segformer' in key):
            new_checkpoint[key] = value
        else:
            print(f"Skipping {key} due to shape mismatch")

    return new_checkpoint




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



def dilate_instances_no_overlap(instances, dilation_kernel=None, iterations=1, dilate = True):
    """
    Dilates each instance in the labeled instance mask without overlapping.

    Args:
        instances (np.ndarray): 2D array of shape (H, W) with instance labels (0 = background).
        dilation_kernel (np.ndarray): Structuring element used for dilation. If None, a 3x3 ellipse is used.
        iterations (int): Number of dilation iterations.

    Returns:
        np.ndarray: New instance mask with dilated instances and no overlap.
    """
    if dilation_kernel is None:
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    output = np.zeros_like(instances, dtype=np.int32)
    occupied_mask = np.zeros_like(instances, dtype=bool)

    instance_ids = np.unique(instances)
    instance_ids = instance_ids[instance_ids != 0]  # Skip background

    for instance_id in instance_ids:
        # Binary mask for current instance
        mask = (instances == instance_id).astype(np.uint8)

        # Dilate the instance
        if dilate:
            dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=iterations).astype(bool)
        else:
            dilated_mask = cv2.erode(mask, dilation_kernel, iterations=iterations).astype(bool)

        # Remove parts that would overlap with already placed instances
        dilated_mask = np.logical_and(dilated_mask, ~occupied_mask)

        # Update output and occupied mask
        output[dilated_mask] = instance_id
        occupied_mask[dilated_mask] = True

    return output

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



def he_to_binary_mask_final(filename,im_pth,inst_pth):
    im_file = filename.replace(inst_pth, im_pth).replace('.xml','.tif')
    xml_file = filename

    # Read XML annotations
    tree = ET.parse(xml_file)
    root = tree.getroot()

    xy = []
    for region in root.iter('Region'):
        vertices = region.find('Vertices')
        coords = []
        for vertex in vertices.findall('Vertex'):
            x = float(vertex.attrib['X'])
            y = float(vertex.attrib['Y'])
            coords.append([x, y])
        xy.append(np.array(coords))

    # Read image dimensions using PIL
    img = Image.open(im_file)
    ncol, nrow = img.size  # width, height
    binary_mask = np.zeros((nrow, ncol), dtype=np.int32)
    color_mask = np.zeros((nrow, ncol, 3), dtype=np.float32)

    for zz, coords in enumerate(xy, 1):
        # print(f'Processing object # {zz}')
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]

        rr, cc = polygon(y_coords, x_coords, shape=binary_mask.shape)
        new_mask = np.zeros_like(binary_mask, dtype=np.int32)
        new_mask[rr, cc] = 1

        binary_mask += zz * (1 - np.minimum(1, binary_mask)) * new_mask
        rand_color = np.random.rand(3)
        for ch in range(3):
            color_mask[rr, cc, ch] += rand_color[ch]

    # Clip color mask to 0–1 range for valid display
    color_mask = np.clip(color_mask, 0, 1)

    # plt.figure()
    # plt.title('Binary Mask')
    # plt.imshow(binary_mask, cmap='nipy_spectral')
    # plt.axis('off')
    #
    # plt.figure()
    # plt.title('Color Mask')
    # plt.imshow(color_mask)
    # plt.axis('off')
    # plt.show()

    return xy, binary_mask, color_mask

def split_patches(arr: np.ndarray, split: int) -> np.ndarray:
    """
    Split [B, W, H, C] into [B * split*split, W//split, H//split, C]

    Args:
        arr: numpy array of shape [B, W, H, C]
        split: number of splits along W and H

    Returns:
        numpy array of shape [B * split**2, W//split, H//split, C]
    """
    try:
        B, W, H, C = arr.shape
    except:
        B, W, H = arr.shape
        C = 1
    assert W % split == 0 and H % split == 0, "W and H must be divisible by split"
    w_new = W // split
    h_new = H // split

    # Step 1: reshape
    arr = arr.reshape(B, split, w_new, split, h_new, C)
    # Step 2: transpose to bring patch axis together
    arr = arr.transpose(0, 1, 3, 2, 4, 5)  # [B, s, s, w_new, h_new, C]
    # Step 3: merge B and s²
    arr = arr.reshape(B * split * split, w_new, h_new, C)
    if C==1:
        arr = arr[:,:,:,0]
    return arr

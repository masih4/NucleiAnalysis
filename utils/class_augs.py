from imgaug import augmenters as iaa
import torch
import kornia.geometry.transform as T
import kornia.augmentation as K
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ThinPlateSplineAugmentation(torch.nn.Module):
    def __init__(self, control_points, std=0.1):
        super().__init__()
        self.control_points = control_points
        self.std = std

    def forward(self, x):
        batch_size, _, height, width = x.shape
        device = x.device

        # Generate random control points perturbation
        perturbation = torch.normal(0, self.std, size=(batch_size, self.control_points, 2), device=device)
        src = T.get_meshgrid2d(self.control_points, self.control_points, normalized_coordinates=True).unsqueeze(0).repeat(batch_size, 1, 1)
        dst = src + perturbation

        # Apply thin-plate spline
        return T.thin_plate_spline(src, dst, x)

def blood_vessel_aug(train_masks = None, train_images = None, num_classes = 6):
    # class_num = 2
    # random.seed(42)
    # sample_inds = random.sample(range(int(np.shape(train_masks)[0]-1)), 20)
    # i_m = np.zeros((np.shape(train_images)[1], np.shape(train_images)[2], np.shape(train_images)[3] + num_classes))
    # counter = 0
    # for i in range(train_images.shape[0]):
    #     if counter >= len(sample_inds) - 1:
    #         break
    #     if np.sum(train_masks[i,:,:] == class_num)>0:
    #         mask = np.copy(train_masks[i,:,:])
    #         mask1 = np.zeros((mask.shape[0], mask.shape[1], num_classes))
    #
    #         for j in range(num_classes):
    #             mask1[:,:,j][mask == j] = mask[mask == j]
    #
    #
    #         image_slice2 = np.copy(train_images[sample_inds[counter],:,:])
    #
    #         i_m[:, :, 0:3] = np.copy(train_images[i])
    #         i_m[:, :, 3:] = np.copy(mask1)
    #
    #         # image_slice[:,:,0] = msk1*image[:,:,0]
    #         # image_slice[:,:,1] = msk1*image[:,:,1]
    #         # image_slice[:,:,2] = msk1*image[:,:,2]
    #
    #         ag1 = iaa.PiecewiseAffine(scale=(0.1),seed=0)
    #         ag2 = iaa.Rot90(1)
    #
    #         i_m = ag1.augment_image(i_m)
    #         i_m = ag2.augment_image(i_m)
    #         mask1 = i_m[:,:,3:]
    #         for j in range(num_classes):
    #             mask1[:,:,j][mask1[:,:,j]>0] = j
    #
    #
    #
    #         msk2 = mask1[:,:,class_num]/class_num
    #
    #         # msk2[msk2>0] = 1
    #
    #         slice_aug = i_m[:,:,0:3]
    #         image_slice2[:,:,0] = (1-msk2)*image_slice2[:,:,0] + msk2*slice_aug[:,:,0]
    #         image_slice2[:,:,1] = (1-msk2)*image_slice2[:,:,1] + msk2*slice_aug[:,:,1]
    #         image_slice2[:,:,2] = (1-msk2)*image_slice2[:,:,2] + msk2*slice_aug[:,:,2]
    #
    #         mask = np.copy(train_masks[sample_inds[counter],:,:])
    #         mask1 = np.zeros((mask.shape[0], mask.shape[1], num_classes))
    #
    #         for j in range(num_classes):
    #             mask1[:,:,j][mask == j] = mask[mask == j]
    #
    #
    #         for j in range(num_classes):
    #             if j == class_num:
    #                 a = mask1[:,:,j]
    #                 a[msk2>0] = j
    #             else:
    #                 a = mask1[:,:,j]
    #                 a[msk2>0] = 0
    #             mask1[:,:,j] = a
    #         dsf = np.sum(mask1, axis=2)
    #         dsf[dsf>num_classes-1] = 0
    #         train_masks[sample_inds[counter],:,:] = dsf
    #         train_images[sample_inds[counter],:,:] = image_slice2
    #         counter += 1
    #

    class_num = 4
    random.seed(43)
    sample_inds = random.sample(range(int(np.shape(train_masks)[0]-1)), 40)
    i_m = np.zeros((np.shape(train_images)[1], np.shape(train_images)[2], np.shape(train_images)[3] + num_classes))
    counter = 0
    for i in range(train_images.shape[0]):
        if counter >= len(sample_inds) - 1:
            break
        if np.sum(train_masks[i,:,:] == class_num)>0:
            mask = np.copy(train_masks[i,:,:])
            mask1 = np.zeros((mask.shape[0], mask.shape[1], num_classes))

            for j in range(num_classes):
                mask1[:,:,j][mask == j] = mask[mask == j]


            image_slice2 = np.copy(train_images[sample_inds[counter],:,:])

            i_m[:, :, 0:3] = np.copy(train_images[i])
            i_m[:, :, 3:] = np.copy(mask1)

            # image_slice[:,:,0] = msk1*image[:,:,0]
            # image_slice[:,:,1] = msk1*image[:,:,1]
            # image_slice[:,:,2] = msk1*image[:,:,2]

            ag1 = iaa.PiecewiseAffine(scale=(0.1),seed=0)
            ag2 = iaa.Rot90(1)

            i_m = ag1.augment_image(i_m)
            i_m = ag2.augment_image(i_m)
            mask1 = i_m[:,:,3:]
            for j in range(num_classes):
                mask1[:,:,j][mask1[:,:,j]>0] = j



            msk2 = mask1[:,:,class_num]/class_num


            # num_labels, label_image = cv2.connectedComponents(msk2.astype(np.uint8))
            #
            # msk3 = np.zeros_like(msk2)

            # msk2[msk2>0] = 1
            # inds = random.sample(range(int(num_labels-1)), 1)[0] + 1
            #
            # msk3[label_image == inds] = 1

            msk3 = msk2
            slice_aug = i_m[:,:,0:3]
            image_slice2[:,:,0] = (1-msk3)*image_slice2[:,:,0] + msk3*slice_aug[:,:,0]
            image_slice2[:,:,1] = (1-msk3)*image_slice2[:,:,1] + msk3*slice_aug[:,:,1]
            image_slice2[:,:,2] = (1-msk3)*image_slice2[:,:,2] + msk3*slice_aug[:,:,2]

            mask = np.copy(train_masks[sample_inds[counter],:,:])
            mask1 = np.zeros((mask.shape[0], mask.shape[1], num_classes))

            for j in range(num_classes):
                mask1[:,:,j][mask == j] = mask[mask == j]


            for j in range(num_classes):
                if j == class_num:
                    a = mask1[:,:,j]
                    a[msk3>0] = j
                else:
                    a = mask1[:,:,j]
                    a[msk3>0] = 0
                mask1[:,:,j] = a
            dsf = np.sum(mask1, axis=2)
            dsf[dsf>num_classes-1] = 0
            train_masks[sample_inds[counter],:,:] = dsf
            train_images[sample_inds[counter],:,:] = image_slice2
            counter += 1




    return train_images, train_masks




def tissue_aug_gpu(train_masks: torch.Tensor, train_images: torch.Tensor, num_classes: int = 6, class_num1 = None):
    """
    Augments train_images and train_masks using Kornia for GPU-accelerated transformations.

    Args:
        train_masks (torch.Tensor): Tensor of shape (N, H, W), containing segmentation masks.
        train_images (torch.Tensor): Tensor of shape (N, C, H, W), containing image data.
        num_classes (int): Number of segmentation classes.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Augmented images and masks.
    """
    device = train_images.device

    # Randomly select indices for augmentation
    sample_inds = random.sample(range(train_masks.size(0)), train_masks.size(0))
    counter = 0
    class_num = class_num1

    for i in range(train_images.size(0)):

        if counter >= int(len(sample_inds)/4):
            break

        # Check if the mask contains the target class
        if np.sum(train_masks[i] == class_num) > 0:
            # Prepare mask tensor
            mask = train_masks[i].clone()  # Convert to NumPy for connected components
            mask1 = torch.zeros((mask.shape[0], mask.shape[1], num_classes), device=device, dtype=torch.long)

            for j in range(num_classes):
                mask1[:, :, j][mask == j] = mask[mask == j]

            # Sample another image for augmentation
            for find_best in range(train_masks.size(0)):
                image_slice2 = train_images[sample_inds[find_best]].clone()
                if (sample_inds[find_best] != i) and abs(image_slice2.median() -  train_images[i].median()) < 0.1:
                    # Combine image and mask
                    i_m = torch.zeros((train_images.size(2), train_images.size(3), image_slice2.shape[0] + num_classes), device=device)
                    i_m[:, :, :image_slice2.shape[0]] = train_images[i].clone().permute(1, 2, 0)  # Convert to HWC
                    i_m[:, :, image_slice2.shape[0]:] = mask1

                    aug = K.AugmentationSequential(
                        K.RandomThinPlateSpline(p=1.0, scale=0.3, same_on_batch = True, keepdim=True),  # Correct parameters
                        K.RandomHorizontalFlip(p=0.5),
                        K.RandomVerticalFlip(p=0.5),
                    )

                    # # Apply Kornia augmentations
                    # aug1 = K.AugmentationSequential(
                    #     K.RandomAffine(degrees=0, translate=(0.0, 0.0), scale=(0.9, 0.95), shear=(0.9, 0.95), p=1.0, resample='nearest'),
                    #     K.RandomHorizontalFlip(p=0.5),
                    # )
                    i_m = i_m.permute(2, 0, 1).unsqueeze(0)  # Convert to NCHW
                    i_m = aug(i_m)
                    i_m = i_m.squeeze(0).permute(1, 2, 0)  # Back to HWC

                    # Separate augmented image and mask
                    slice_aug = i_m[:, :, :image_slice2.shape[0]]
                    mask1_aug = i_m[:, :, image_slice2.shape[0]:]

                    if mask1_aug[:,:,class_num].sum().item() > 0:

                        for j in range(num_classes):
                            mask1_aug[:, :, j][mask1_aug[:, :, j] > 0] = j

                        msk2 = mask1_aug[:, :, class_num].cpu().numpy() / class_num

                        # Connected components
                        num_labels, label_image,st1,st2 = cv2.connectedComponentsWithStats(msk2.astype(np.uint8))
                        msk3 = np.zeros_like(msk2)
                        msk5 = np.zeros_like(msk2)
                        msk4 = np.zeros_like(msk2)

                        for mp in range(1,num_labels):
                            msk4[msk2 ==mp] = 1
                            whole_area = (msk4.shape[0]*msk4.shape[1])
                            if np.sum(msk4>0):
                                if np.sum(msk4) < 0.2* whole_area:
                                    msk5 += msk4
                                    break
                                else:
                                    msk3 = 0 * msk3
                                    rand_scale = 2#int(np.ceil((100 * np.sum(msk2) / (msk3.shape[0] * msk3.shape[0]))))

                                    inddr = random.sample(range(rand_scale), 1)
                                    inddc = random.sample(range(rand_scale), 1)
                                    current_label_stats = st1[mp]
                                    label_row_lent = int((current_label_stats[3])/rand_scale)
                                    label_col_lent = int((current_label_stats[2])/rand_scale)
                                    row_range = current_label_stats[1] + inddr[0]*label_row_lent
                                    col_range = current_label_stats[0] + inddc[0]*label_col_lent
                                    msk3[row_range:row_range + label_row_lent -1
                                        ,col_range: col_range + label_col_lent - 1] = 1
                                    msk4 = msk4 * msk3
                                    msk5 += msk4
                                    break
                    #
                    # msk5 = msk2

                    #     if np.sum(msk4) > 0 and (class_num == 5):
                        #         msk3 = 0 * msk3
                        #         rand_scale = 2#int(np.ceil((100 * np.sum(msk2) / (msk3.shape[0] * msk3.shape[0]))))
                        #
                        #         inddr = random.sample(range(rand_scale), 1)
                        #         inddc = random.sample(range(rand_scale), 1)
                        #         current_label_stats = st1[mp]
                        #         label_row_lent = int((current_label_stats[3])/rand_scale)
                        #         label_col_lent = int((current_label_stats[2])/rand_scale)
                        #         row_range = current_label_stats[1] + inddr[0]*label_row_lent
                        #         col_range = current_label_stats[0] + inddc[0]*label_col_lent
                        #         msk3[row_range:row_range + label_row_lent -1
                        #             ,col_range: col_range + label_col_lent - 1] = 1
                        #         msk4 = msk4 * msk3
                        #         msk5 += msk4
                        #     msk5[msk5>0] = 1
                        #
                        # msk5 = msk2



                        # # Randomly pick a connected component
                        # try:
                        #     inds = random.sample(range(1,num_labels), 1)[0]
                        # except:
                        #     continue# Avoid background (label 0)
                        # msk3[label_image == inds] = 1
                        if np.sum(msk5) > 0:
                            msk3 = msk5
                            # Convert msk3 back to PyTorch
                            msk3 = torch.from_numpy(msk3).float().to(device)

                            # Blend augmented slice into the sampled image
                            for kjm in range(image_slice2.shape[0]):
                                image_slice2[kjm] = (1 - msk3) * image_slice2[kjm] + msk3 * slice_aug[:, :, kjm]


                            # Update mask
                            mask = train_masks[sample_inds[counter]].clone()
                            mask1 = torch.zeros((mask.size(0), mask.size(1), num_classes), device=device, dtype=torch.long)

                            for j in range(num_classes):
                                mask1[:, :, j][mask == j] = mask[mask == j]

                            for j in range(num_classes):
                                if j == class_num:
                                    a = mask1[:, :, j]
                                    a[msk3 > 0] = j
                                else:
                                    a = mask1[:, :, j]
                                    a[msk3 > 0] = 0
                                mask1[:, :, j] = a

                            dsf = mask1.sum(dim=2)
                            dsf[dsf > num_classes - 1] = 0
                            # plt.imshow(image_slice2.permute([1, 2, 0]).cpu())
                            # plt.show()
                            # plt.imshow(dsf.cpu())
                            # plt.show()
                            # Update train_images and train_masks
                            train_masks[sample_inds[counter]] = dsf
                            train_images[sample_inds[counter]] = image_slice2
                            counter += 1

    return train_images, train_masks

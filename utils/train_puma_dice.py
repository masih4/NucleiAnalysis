from transformers import SegformerImageProcessor
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from utils.utils_puma import Mine_resize, puma_f1_loss_custom
from utils.utils_puma import KorniaAugmentation
from torch import optim
from utils.LoadPumaData import PumaTissueDataset
from utils.utils_puma import collate_tile_patches
from torch.cuda.amp import autocast, GradScaler
import torch.nn.utils
from tqdm import tqdm
from utils.utils_puma import dice_loss_binary
# from src.utils.kd_loss import MSELoss
from torch.utils.data import DataLoader
import copy
from Models.CellVit.CellViT.cell_segmentation.utils.post_proc_cellvit import DetectionCellPostProcessor
from utils.utils_nuclei import inst_loss_hovernext as criterion_hovernext
# from Models.CellVit.CellViT.base_ml.base_loss import MSGELossMaps
import matplotlib.pyplot as plt
import torchstain
import numpy as np
from Models.CellVit.CellViT.cell_segmentation.utils.metrics import get_fast_pq, remap_label
from Models.CellVit.CellViT.base_ml.base_loss import FocalTverskyLoss

def freeze_enc(model):
    for p in model.encoder.parameters():
        p.requires_grad = False


def unfreeze_enc(model):
    for p in model.encoder.parameters():
        p.requires_grad = True




def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
        target_siz = (128,128),
        n_class = 6,
        image_data1 = None,
        mask_data1 = None,
        val_images = None,
        val_masks = None,
        class_weights = torch.ones(6),
        augmentation = True,
        val_batch = 1,
    early_stopping = 8,
        ful_size = (1024,1024),
        phase_mode = ['train', 'val'],
        dir_checkpoint = Path('E:/PumaDataset/checkpoints/'),
        er_di = False,
        model_name = '',
        val_sleep_time = 0,
        nuclei = False,
        segformer_variant = "nvidia/segformer-b2-finetuned-ade-512-512",
        stain_norm=False,
):
    # 1. Create dataset

    train_set = PumaTissueDataset(image_data1,
                                      mask_data1,
                                      n_class1=n_class,
                                      size1=target_siz,
                                    device1=device,
                                      transform = augmentation,
                                  target_size=ful_size,
                                  # train_indexes = train_indexes,
                                  mode='train',
                                  er_di = er_di,
                                  stain_norm=stain_norm,)
    if val_images is not None:
        val_set = PumaTissueDataset(val_images,
                                          val_masks,
                                          n_class1=n_class,
                                          size1=target_siz,
                                        device1=device,
                                          transform = None,
                                    target_size=ful_size,
                                    mode='valid',
                                    er_di = er_di)

    n_train = len(train_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0)#os.cpu_count(), pin_memory=True)
    val_loader_args = dict(batch_size=val_batch, num_workers=0)#os.cpu_count(), pin_memory=True)
    aug_pipeline = KorniaAugmentation(
        mode="train", num_classes=n_class, seed=42, size=target_siz
    )


    if val_images is not None:
        dataloaders = {
            'train': DataLoader(train_set,shuffle=True,collate_fn=collate_tile_patches, **loader_args),
            'val': DataLoader(val_set, shuffle=False, drop_last=False,collate_fn=collate_tile_patches, **val_loader_args),
        }
    else :
        dataloaders = {
            'train': DataLoader(train_set, shuffle=True,collate_fn=collate_tile_patches, **loader_args),
        }
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Default optimizer setup
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=15,factor=weight_decay,min_lr=0.5*1e-7,cooldown=5)  # goal: maximize Dice score

    optimizer.zero_grad()
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_hsv = nn.MSELoss()
    validation_pq = DetectionCellPostProcessor()
    criterion_focal = FocalTverskyLoss()
    # criterionMSGE = MSGELossMaps()
    # 5. Begin training
    counter = 0
    random.seed(42)

    best_val_score = 0
    for epoch in range(1, epochs + 1):
        if counter > early_stopping:
            break
        scaler = GradScaler(enabled=amp)
        gradient_clipping = 1.0  # Gradient clipping value
        for phase in phase_mode:
            if phase == 'train':
                model.train()
                epoch_loss = 0
                f1_dice = 0
                epoch_dice = torch.zeros(n_class, device=device)
                with (tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar):
                    for images, true_masks in dataloaders[phase]:
                        # images, true_masks = Mine_resize(image=images, mask=true_masks, final_size=target_siz)
                        # Move to device


                        images = images.to(device=device, dtype=torch.float32)


                        true_masks = true_masks.to(device=device, dtype=torch.float32)
                        aug_num = random.choice([0,1,2,3,5,6,7,8,9,10])
                        if aug_num > 2:
                            if augmentation:
                                images, true_masks = aug_pipeline(image=images, mask=true_masks)

                        # torch.clamp(images, 0, 1)
                        # true_masks = true_masks.long()
                        optimizer.zero_grad()
                        # Mixed Precision Training
                        with autocast(enabled=amp):
                            if model_name == 'hovernext':
                                if epoch < 25:
                                    freeze_enc(model)
                                else:
                                    unfreeze_enc(model)

                            if model_name == 'acs':
                                if epoch < 25:
                                    for seg_temp in model.segformer.parameters():
                                            seg_temp.requires_grad = False

                                    for unet_temp in model.unet_encoder.parameters():
                                        unet_temp.requires_grad = False
                                else:

                                    for seg_temp in model.segformer.parameters():
                                        seg_temp.requires_grad = True

                                    for unet_temp in model.unet_encoder.parameters():
                                        unet_temp.requires_grad = True

                            masks_pred = model.forward(images)

                        if model_name == 'hovernext':
                            loss_inst = criterion_hovernext(masks_pred[:,:5], true_masks)
                            loss_binary = criterion_binary(masks_pred[:,5], true_masks[:,1])
                            loss_dice = dice_loss_binary(masks_pred[:,5], true_masks[:,1])
                            binary_pred = masks_pred[:,5]
                            loss = 0.33*loss_inst + 0.33*loss_binary + 0.33*loss_dice
                        # elif model_name == 'cellvit':
                        #     loss_inst = 2.5*criterion_hsv(masks_pred['hv_map'], true_masks[:,2:4])
                        #     loss_msge = 8*criterionMSGE(masks_pred['hv_map'], true_masks[:,2:4],focus=true_masks[:,1],
                        # device=device,)
                        #     loss_binary = criterion_focal(masks_pred['nuclei_binary_map'], true_masks[:,1])
                        #     loss_dice = dice_loss_binary(masks_pred['nuclei_binary_map'], true_masks[:,1])
                        #     binary_pred = torch.argmax(masks_pred['nuclei_binary_map'], dim=1)
                        #
                        #     loss = 0.33*loss_inst + 0.33*loss_binary + 0.33*loss_dice + 0.33*loss_msge
                        elif model_name == 'acs':
                            loss_inst = criterion_hovernext(masks_pred[:,:5], true_masks)
                            # loss_binary = criterion_focal(masks_pred[:,5:7], true_masks[:,1])
                            loss_binary = criterion_binary(masks_pred[:,5], true_masks[:,1])
                            loss_dice = dice_loss_binary(masks_pred[:,5], true_masks[:,1])
                            binary_pred = masks_pred[:,5]
                            loss = 0.33*loss_inst + 0.33*loss_binary + 0.33*loss_dice
                        scaler.scale(loss).backward()

                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                        # Optimizer step
                        scaler.step(optimizer)
                        scaler.update()

                        # Update progress
                        epoch_loss += loss.item() * true_masks.size(0)  # Accumulate loss weighted by batch size
                        pbar.update(true_masks.shape[0])
                        # if model_name == 'hovernext':
                        masks_pred = F.sigmoid(binary_pred)
                        masks_pred = (masks_pred > 0.5).float()
                        # elif model_name == 'cellvit':
                        #     masks_pred = torch.argmax(binary_pred, dim=1).float()
                        # else:
                        #     print('hi')
                        # Calculate dice scores
                        with torch.no_grad():
                            preds_class = masks_pred
                            true_class = (true_masks[:,1]).float()
                            intersection = (preds_class * true_class).sum()
                            union = preds_class.sum() + true_class.sum()
                            dice_scores = (2 * intersection + 1e-7) / (union + 1e-7)
                            epoch_dice += dice_scores * true_masks.size(0)  # Accumulate weighted Dice
            if phase == 'train':
                epoch_loss /= n_train  # Divide by total training samples
                epoch_dice /= n_train
                print(epoch_dice.mean(), epoch_loss)
            elif phase == 'val' and epoch> val_sleep_time:  # Validation phase
                model.eval()
                total_val_images = 0  # Track total validation images
                val_pq = []
                with torch.no_grad():
                    for images, true_masks in dataloaders[phase]:
                        # images, true_masks = Mine_resize(image=images, mask=true_masks, final_size=target_siz)
                        images = images.to(device=device, dtype=torch.float32)
                        true_masks = true_masks.to(device=device, dtype=torch.float32)
                        with autocast(enabled=amp):
                            masks_pred = model.forward(images)
                            if model_name == 'hovernext':
                                binary_pred = masks_pred[:, 5]
                                binary_pred = (F.sigmoid(binary_pred) > 0.5).float()
                                preds = torch.cat([binary_pred.unsqueeze(1), masks_pred[:,0:2]], dim=1)
                                preds = np.transpose(preds.cpu().numpy(),(0,2,3,1))
                            # elif model_name == 'cellvit':
                            #     binary_pred = torch.argmax(masks_pred['nuclei_binary_map'], dim=1)
                            #     preds = masks_pred['hv_map'].permute(0,2,3,1).cpu().numpy()
                                # preds = torch.cat([binary_pred.unsqueeze(1), masks_pred['hv_map']], dim=1)
                                # preds = np.transpose(preds.cpu().numpy(),(0,2,3,1))
                            elif model_name == 'acs':
                                binary_pred = masks_pred[:, 5]
                                binary_pred = (F.sigmoid(binary_pred) > 0.5).float()
                                preds = torch.cat([binary_pred.unsqueeze(1), masks_pred[:,0:2]], dim=1)
                                preds = np.transpose(preds.cpu().numpy(),(0,2,3,1))
                            pred_inst,_ = validation_pq.post_process_cell_segmentation(pred_map=preds)
                            pred_inst = remap_label(pred_inst)
                            gt = remap_label(true_masks[0,0].cpu().numpy().astype(np.int32))
                            pq,_ = get_fast_pq(gt, pred_inst)
                            val_pq.append(pq[0])
                    # print('new metrics: ', total_dice, total_iou)
                    val_pq = np.mean(np.stack(val_pq))  # Divide by total validation samples
                    print(
                        f"pq: {val_pq:.4f}")
                    th = val_pq
                    scheduler.step(th)
                    counter += 1
                    if th > best_val_score:
                        counter = 0
                        best_val_score = th
                        print( 'saving best model')
                        print(
                            f"pq: {val_pq:.4f}")
                        if save_checkpoint:
                            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                            state_dict = model.state_dict()
                            # state_dict['mask_values'] = dataset.mask_values
                            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(1)))
                            best_model_wts = copy.deepcopy(state_dict)

    try:
        model.load_state_dict(best_model_wts)
    except:
        print('no model loaded')
    return model


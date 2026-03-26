# -*- coding: utf-8 -*-
# Running an Experiment Using CellViT cell segmentation network
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import inspect
import os
import sys
from Models.CellVit.CellViT.cell_segmentation.utils.post_proc_cellvit import DetectionCellPostProcessor
from utilsHover.utils_nuclei import get_fast_dice_2, get_fast_aji,gen_instance_hv_maps

from debugpy.common.log import log_dir
from Models.CellVit.CellViT.cell_segmentation.utils.metrics import get_fast_pq, remap_label
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import wandb
from DataSetTools.utils1 import train_val_split
from base_ml.base_cli import ExperimentBaseParser
from cell_segmentation.experiments.experiment_cellvit_pannuke import (
    ExperimentCellVitPanNuke,
)
from cell_segmentation.experiments.experiment_cellvit_conic import (
    ExperimentCellViTCoNic,
)

from cell_segmentation.inference.inference_cellvit_experiment_pannuke import (
    InferenceCellViT,
)
import torch
import numpy as np
import random
import cv2
from torchvision import transforms as T


mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
inference_transforms_vit = T.Compose(
            [T.ToTensor(), T.Normalize(mean=mean, std=std)]
        )

def evalAndSave(dir_checkpoint = None, model = None, fold = None, dataPath = None, device = None):
    model.eval()
    model.to(device)
    with torch.no_grad():
        kk = 0
        results_dict = {}

        results_dict['PQ'] = []
        results_dict['DQ'] = []
        results_dict['SQ'] = []
        results_dict['precision'] = []
        results_dict['recall'] = []
        results_dict['AJI'] = []
        results_dict['DICE'] = []
        validation_pq = DetectionCellPostProcessor()
        SQ = []
        DQ = []
        PQ = []
        precision = []
        recall = []
        AJI = []
        DICE = []

        imPth = os.path.join(dataPath,'fold1','images')
        mskPth = os.path.join(dataPath,'fold1','labels')
        ims = []
        msk = []
        for im in os.listdir(imPth):
            ims.append(os.path.join(imPth,im))
            msk.append(os.path.join(mskPth,im.replace('.tif','.npy')))


        for images, true_masks in zip(ims, msk):
            mask_name = true_masks
            images = cv2.imread(images)
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            true_masks = np.load(true_masks)


            true_masks = gen_instance_hv_maps(true_masks[np.newaxis, ...])


            images = inference_transforms_vit(images).unsqueeze(0)
            true_masks = true_masks.astype("float32")
            true_masks = torch.tensor(true_masks, dtype=torch.float32)



            # images, true_masks = Mine_resize(image=images, mask=true_masks, final_size=target_siz)
            images = images.to(device=device, dtype=torch.float32)
            masks_pred = model.forward(images)
            binary_pred = torch.argmax(masks_pred['nuclei_binary_map'], dim=1)
            preds = torch.cat([binary_pred.unsqueeze(1), masks_pred['hv_map']], dim=1)
            preds = np.transpose(preds.cpu().numpy(),(0,2,3,1))
            pred_inst, _ = validation_pq.post_process_cell_segmentation(pred_map=preds)
            pred_inst = remap_label(pred_inst)
            os.makedirs(os.path.join(dir_checkpoint, f'fold{fold}Result'), exist_ok=True)
            # save_dir = os.path.join(dir_checkpoint,f'fold{fold}Result', val_names[kk].replace('.tif','.png'))
            # cv2.imwrite(save_dir,pred_inst)
            if np.max(pred_inst) < 255:
                pred_inst = pred_inst.astype(np.uint8)
            else:
                pred_inst = pred_inst.astype(np.uint16)
            save_dir = os.path.join(dir_checkpoint, f'fold{fold}Result', mask_name.split('/')[-1])
            np.save(save_dir, pred_inst)
            kk += 1

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
        SQ = np.mean(np.stack(SQ))
        DQ = np.mean(np.stack(DQ))
        PQ = np.mean(np.stack(PQ))
        precision = np.mean(np.stack(precision))
        recall = np.mean(np.stack(recall))
        aji = np.mean(np.stack(AJI))
        dice = np.mean(np.stack(DICE))
        results_dict['PQ'] = PQ
        results_dict['DQ'] = DQ
        results_dict['SQ'] = SQ
        results_dict['precision'] = precision
        results_dict['recall'] = recall
        results_dict['AJI'] = aji
        results_dict['DICE'] = dice
        pthh = os.path.join(dir_checkpoint, 'results.npy')
        np.save(pthh, results_dict, allow_pickle=True)


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

if __name__ == "__main__":
    # Parse arguments
    for fold in range(5):
        conf_pth = '/home/ntorbati/PycharmProjects/NucleiAnalysis/Models/CellVit/CellViT/configs/nucleiAnalysis'
        configuration_parser = ExperimentBaseParser()
        configuration = configuration_parser.parse_arguments()


        data_path = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/custom_split/NuInsSeg/train/'
        logs_dir = '/home/ntorbati/PycharmProjects/NucleiAnalysis/Models/CellVit/CellViT/logs_paper/logs/'
        configuration['random_seed'] = 19
        seed_torch(configuration['random_seed'])
        configuration['data']['input_shape'] = 512
        configuration['training']['epochs'] = 150

        train_val_split(data_path=data_path, random_state=configuration['random_seed'], fold=fold, conf_pth=conf_pth)

        configuration['data']['dataset_path'] = os.path.join(data_path,'train')
        wandb_dir = os.path.join(logs_dir,'fold' , str(fold),'train')
        os.makedirs(wandb_dir, exist_ok=True)
        configuration['logging']['wandb_dir'] = wandb_dir

        log_dir = os.path.join(logs_dir,'fold', str(fold) ,'train')
        os.makedirs(log_dir, exist_ok=True)
        configuration['logging']['log_dir'] = log_dir


        if configuration["data"]["dataset"].lower() == "pannuke":
            experiment_class = ExperimentCellVitPanNuke
        elif configuration["data"]["dataset"].lower() == "conic":
            experiment_class = ExperimentCellViTCoNic
        # Setup experiment
        if "checkpoint" in configuration:
            # continue checkpoint
            experiment = experiment_class(
                default_conf=configuration, checkpoint=configuration["checkpoint"]
            )
            outdir = experiment.run_experiment()
            inference = InferenceCellViT(
                run_dir=outdir,
                gpu=configuration["gpu"],
                checkpoint_name=configuration["eval_checkpoint"],
                magnification=configuration["data"].get("magnification", 40),
            )
            (
                trained_model,
                inference_dataloader,
                dataset_config,
            ) = inference.setup_patch_inference()
            inference.run_patch_inference(
                trained_model, inference_dataloader, dataset_config, generate_plots=False
            )
        else:
            experiment = experiment_class(default_conf=configuration)
            if configuration["run_sweep"] is True:
                # run new sweep
                sweep_configuration = experiment_class.extract_sweep_arguments(
                    configuration
                )
                os.environ["WANDB_DIR"] = os.path.abspath(
                    configuration["logging"]["wandb_dir"]
                )
                sweep_id = wandb.sweep(
                    sweep=sweep_configuration, project=configuration["logging"]["project"]
                )
                wandb.agent(sweep_id=sweep_id, function=experiment.run_experiment)
            elif "agent" in configuration and configuration["agent"] is not None:
                # add agent to already existing sweep, not run sweep must be set to true
                configuration["run_sweep"] = True
                os.environ["WANDB_DIR"] = os.path.abspath(
                    configuration["logging"]["wandb_dir"]
                )
                wandb.agent(
                    sweep_id=configuration["agent"], function=experiment.run_experiment
                )
            else:
                # casual run
                outdir = experiment.run_experiment()
                inference = InferenceCellViT(
                    run_dir=outdir,
                    gpu=configuration["gpu"],
                    checkpoint_name=configuration["eval_checkpoint"],
                    magnification=configuration["data"].get("magnification", 40),
                )
                (
                    trained_model,
                    inference_dataloader,
                    dataset_config,
                ) = inference.setup_patch_inference()
                evalAndSave(dir_checkpoint=outdir, model=trained_model,
                            dataPath=configuration['data']['dataset_path'], device=configuration["gpu"],
                            fold = fold)

                # inference.run_patch_inference(
                #     trained_model,
                #     inference_dataloader,
                #     dataset_config,
                #     generate_plots=False,
                # )
        wandb.finish()

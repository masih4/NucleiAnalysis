# -*- coding: utf-8 -*-
# Running an Experiment Using CellViT cell segmentation network
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import inspect
import os
import sys
import shutil
from debugpy.common.log import log_dir

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import wandb

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


def copy_with_rename(src, dst, index):
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        # Create corresponding directory
        rel_path = os.path.relpath(root, src)
        dest_dir = os.path.join(dst, rel_path)
        os.makedirs(dest_dir, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)

            # rename using constant index
            name, ext = os.path.splitext(file)
            new_name = f"{name}_{index}{ext}"

            dest_file = os.path.join(dest_dir, new_name)

            shutil.copy2(src_file, dest_file)

def copy_ims(joinBest,dest_folder1,dest_folder2):
    dataset_tags = ['pan-cancer-nuclei-seg', 'puma', 'monuseg', 'cpm17', 'tnbc' , 'nuinsseg',
                    'consep','monusac', 'data science bowl', 'cryonuseg'][0:joinBest]
    data_path = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/'
    # logs_dir = '/home/ntorbati/PycharmProjects/NucleiAnalysis/Models/CellVit/CellViT/logs_paper/logs'
    datasets_names = os.listdir(data_path)
    for ind,dataset_tag in zip(range(joinBest) , dataset_tags):
        for dataset_name in datasets_names:
            if dataset_tag in dataset_name:
                dataset_path = os.path.join(data_path, dataset_name, 'train', 'fold0')
                validation_path = os.path.join(data_path, dataset_name, 'train', 'fold1')

                for pth in ['images','labels']:
                    copy_with_rename(os.path.join(dataset_path,pth), os.path.join(dest_folder1,pth),ind)
                    copy_with_rename(os.path.join(validation_path, pth), os.path.join(dest_folder2, pth), ind)


                # shutil.copytree(dataset_path, dest_folder1,dirs_exist_ok=True)
                # shutil.copytree(validation_path, dest_folder2,dirs_exist_ok=True)


if __name__ == "__main__":
    # Parse arguments
    configuration_parser = ExperimentBaseParser()
    configuration = configuration_parser.parse_arguments()
    data_path = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/'

    for joinBest in [2,3,4,5,6,7,8,9,10]:
        dataset_tags = ['trainAll' + str(joinBest)]
        dest_folder1 = os.path.join(data_path, dataset_tags[0], 'train', 'fold0')
        dest_folder2 = os.path.join(data_path, dataset_tags[0], 'train', 'fold1')
        os.makedirs(dest_folder1, exist_ok=True)
        os.makedirs(dest_folder2, exist_ok=True)
        copy_ims(joinBest, dest_folder1,dest_folder2)





        logs_dir = '/home/ntorbati/PycharmProjects/NucleiAnalysis/Models/CellVit/CellViT/logs_paper/logs'
        datasets_names = os.listdir(data_path)
        for dataset_tag in dataset_tags:
            for dataset_name in datasets_names:
                if dataset_tag in dataset_name:
                    print(dataset_name)
                    configuration['data']['dataset_path'] = os.path.join(data_path,dataset_name,'train')
                    wandb_dir = os.path.join(logs_dir,dataset_name,'train')
                    os.makedirs(wandb_dir, exist_ok=True)
                    configuration['logging']['wandb_dir'] = wandb_dir

                    log_dir = os.path.join(logs_dir,dataset_name,'train')
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
                            # inference.run_patch_inference(
                            #     trained_model,
                            #     inference_dataloader,
                            #     dataset_config,
                            #     generate_plots=False,
                            # )
                    wandb.finish()

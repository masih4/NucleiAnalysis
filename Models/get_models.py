import inspect
import os
import sys

import yaml

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from Models.CellVit.CellViT.utils.tools import unflatten_dict
from pathlib import Path
from typing import Callable, Tuple, Union
import torch
from Models.CellVit.CellViT.models.segmentation.cell_segmentation.cellvit import (
    CellViT,
    CellViTSAM,
    CellViT256,
)
from Models.CellVit.CellViT.models.segmentation.cell_segmentation.cellvit_shared import (
    CellViTShared,
    CellViT256Shared,
    CellViTSAMShared,
)

from Models.HoverNet.hover_net.models.hovernet.net_desc import HoVerNet

import matplotlib.pyplot as plt

from Models.HoverNet.hover_net.models.hovernet.post_proc import process
from Models.HoverNext.hover_next_train.src.multi_head_unet import get_model





def get_train_model(
        self,
        pretrained_encoder: Union[Path, str] = None,
        pretrained_model: Union[Path, str] = None,
        backbone_type: str = "vit256",
        shared_decoders: bool = False,
        regression_loss: bool = False,
        **kwargs,
) -> CellViT:
    """Return the CellViT training model

    Args:
        pretrained_encoder (Union[Path, str]): Path to a pretrained encoder. Defaults to None.
        pretrained_model (Union[Path, str], optional): Path to a pretrained model. Defaults to None.
        backbone_type (str, optional): Backbone Type. Currently supported are default (None, ViT256, SAM-B, SAM-L, SAM-H). Defaults to None
        shared_decoders (bool, optional): If shared skip decoders should be used. Defaults to False.
        regression_loss (bool, optional): If regression loss is used. Defaults to False

    Returns:
        CellViT: CellViT training model with given setup
    """
    # reseed needed, due to subprocess seeding compatibility

    # check for backbones
    implemented_backbones = ["default", "vit256", "sam-b", "sam-l", "sam-h"]
    if backbone_type.lower() not in implemented_backbones:
        raise NotImplementedError(
            f"Unknown Backbone Type - Currently supported are: {implemented_backbones}"
        )
    if backbone_type.lower() == "default":
        if shared_decoders:
            model_class = CellViTShared
        else:
            model_class = CellViT
        model = model_class(
            num_nuclei_classes=self["data"]["num_nuclei_classes"],
            num_tissue_classes=self["data"]["num_tissue_classes"],
            embed_dim=self["model"]["embed_dim"],
            input_channels=self["model"].get("input_channels", 3),
            depth=self["model"]["depth"],
            num_heads=self["model"]["num_heads"],
            extract_layers=self["model"]["extract_layers"],
            drop_rate=self["training"].get("drop_rate", 0),
            attn_drop_rate=self["training"].get("attn_drop_rate", 0),
            drop_path_rate=self["training"].get("drop_path_rate", 0),
            regression_loss=regression_loss,
        )

        if pretrained_model is not None:
            logger.info(
                f"Loading pretrained CellViT model from path: {pretrained_model}"
            )
            cellvit_pretrained = torch.load(pretrained_model)
            logger.info(model.load_state_dict(cellvit_pretrained, strict=True))
            logger.info("Loaded CellViT model")

    if backbone_type.lower() == "vit256":
        if shared_decoders:
            model_class = CellViT256Shared
        else:
            model_class = CellViT256
        model = model_class(
            model256_path=pretrained_encoder,
            num_nuclei_classes=self["data"]["num_nuclei_classes"],
            num_tissue_classes=self["data"]["num_tissue_classes"],
            drop_rate=self["training"].get("drop_rate", 0),
            attn_drop_rate=self["training"].get("attn_drop_rate", 0),
            drop_path_rate=self["training"].get("drop_path_rate", 0),
            regression_loss=regression_loss,
        )
        # model.model256_path = '/home/ntorbati/PycharmProjects/NucleiAnalysis/Models/CellVit/CellViT/weights/vit256_small_dino.pth'
        # model.load_pretrained_encoder(model.model256_path)
        if pretrained_model:
            # pretrained_model = '/home/ntorbati/PycharmProjects/NucleiAnalysis/Models/CellVit/CellViT/logs_paper/logs/2025-10-31T144629_None/checkpoints/model_best.pth'
            cellvit_pretrained = torch.load(pretrained_model, map_location="cpu")
            run_conf = unflatten_dict(cellvit_pretrained["config"], ".")
            model = get_model_new(run_conf, model_type=cellvit_pretrained["arch"])

            # logger.info(
            #     f"Loading pretrained CellViT model from path: {pretrained_model}"
            # )

            model.load_state_dict(cellvit_pretrained["model_state_dict"])
        # model.model256_path = '/home/ntorbati/PycharmProjects/NucleiAnalysis/Models/CellVit/CellViT/weights/vit256_small_dino.pth'
        # model.load_pretrained_encoder(model.model256_path)

        # model.freeze_encoder()
        # logger.info("Loaded CellVit256 model")
    if backbone_type.lower() in ["sam-b", "sam-l", "sam-h"]:
        if shared_decoders:
            model_class = CellViTSAMShared
        else:
            model_class = CellViTSAM
        model = model_class(
            model_path=pretrained_encoder,
            num_nuclei_classes=self["data"]["num_nuclei_classes"],
            num_tissue_classes=self["data"]["num_tissue_classes"],
            vit_structure=backbone_type,
            drop_rate=self["training"].get("drop_rate", 0),
            regression_loss=regression_loss,
        )
        model.model_path = '/home/ntorbati/PycharmProjects/NucleiAnalysis/Models/CellVit/CellViT/weights/sam_vit_l.pth'
        model.load_pretrained_encoder(model.model_path)
        pretrained_model = '/home/ntorbati/PycharmProjects/NucleiAnalysis/Models/CellVit/CellViT/weights/sam_vit_l.pth'
        if pretrained_model is not None:
            # logger.info(
            #     f"Loading pretrained CellViT model from path: {pretrained_model}"
            # )
            cellvit_pretrained = torch.load(pretrained_model, map_location="cpu")
            model.load_state_dict(cellvit_pretrained, strict=True)
            # logger.info(model.load_state_dict(cellvit_pretrained, strict=True))
        model.freeze_encoder()
        # logger.info(f"Loaded CellViT-SAM model with backbone: {backbone_type}")

    # logger.info(f"\nModel: {model}")
    # model = model.to("cpu")

    return model


def get_model_new(
    run_conf, model_type: str
) -> Union[
    CellViT,
    CellViTShared,
    CellViT256,
    CellViT256Shared,
    CellViTSAM,
    CellViTSAMShared,
]:
    """Return the trained model for inference

    Args:
        model_type (str): Name of the model. Must either be one of:
            CellViT, CellViTShared, CellViT256, CellViT256Shared, CellViTSAM, CellViTSAMShared

    Returns:
        Union[CellViT, CellViTShared, CellViT256, CellViTShared, CellViTSAM, CellViTSAMShared]: Model
    """
    implemented_models = [
        "CellViT",
        "CellViTShared",
        "CellViT256",
        "CellViT256Shared",
        "CellViTSAM",
        "CellViTSAMShared",
    ]
    if model_type not in implemented_models:
        raise NotImplementedError(
            f"Unknown model type. Please select one of {implemented_models}"
        )
    if model_type in ["CellViT", "CellViTShared"]:
        if model_type == "CellViT":
            model_class = CellViT
        elif model_type == "CellViTShared":
            model_class = CellViTShared
        model = model_class(
            num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            embed_dim=run_conf["model"]["embed_dim"],
            input_channels=run_conf["model"].get("input_channels", 3),
            depth=run_conf["model"]["depth"],
            num_heads=run_conf["model"]["num_heads"],
            extract_layers=run_conf["model"]["extract_layers"],
            regression_loss=run_conf["model"].get("regression_loss", False),
        )

    elif model_type in ["CellViT256", "CellViT256Shared"]:
        if model_type == "CellViT256":
            model_class = CellViT256
        elif model_type == "CellViT256Shared":
            model_class = CellViT256Shared
        model = model_class(
            model256_path=None,
            num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            regression_loss=run_conf["model"].get("regression_loss", False),
        )
    elif model_type in ["CellViTSAM", "CellViTSAMShared"]:
        if model_type == "CellViTSAM":
            model_class = CellViTSAM
        elif model_type == "CellViTSAMShared":
            model_class = CellViTSAMShared
        model = model_class(
            model_path=None,
            num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            vit_structure=run_conf["model"]["backbone"],
            regression_loss=run_conf["model"].get("regression_loss", False),
        )
    return model


if __name__ == '__main__':
    dataset_config_path = "/home/ntorbati/PycharmProjects/NucleiAnalysis/configs/train_cellvit_256.yaml"
    with open(dataset_config_path, "r") as dataset_config_file:
        yaml_config = yaml.safe_load(dataset_config_file)
        dataset_config = dict(yaml_config)
    model = get_train_model(dataset_config)
    # print('model')
    x = torch.randn(1, 3, 1024, 1024).to('cuda')
    # model.to('cuda')

    # model1 = HoVerNet(nr_types=None)

    model2 = get_model(out_channels_cls=8,
    out_channels_inst=5,
    pretrained=True,)

    # model1.to('cuda')

    model2.to('cuda')

    with torch.no_grad():
        # out = model(x)
        out2 = model2(x)
        process(out2)
        print("Output shape:", tuple(out2.shape))

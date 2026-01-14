import numpy as np
from matplotlib import pyplot as plt

from Models.CellVit.CellViT.cell_segmentation.utils.tools import get_bounding_box
from scipy.ndimage import center_of_mass, distance_transform_edt
from Models.CellVit.CellViT.cell_segmentation.utils.post_proc_cellvit import DetectionCellPostProcessor
import mahotas as mh
import torch.nn.functional as F
import torch

def inst_loss_hovernext(input, gt):

    loss_cpv = F.mse_loss(input=input[:, :2], target=gt[:,2:4])
    loss_3c = F.cross_entropy(
        input=input[:, 2:5],
        target=gt[:,4].long(),
        weight=torch.tensor([1, 1, 2]).type_as(input).to(input.device),
    )
    return loss_cpv + loss_3c


def inst_to_3c(gt_inst):
    borders = mh.labeled.borders(gt_inst, Bc=np.ones((3, 3)))
    mask = gt_inst > 0
    return (((borders & mask) * 1) + (mask * 1))[np.newaxis, :]

def make_cpvs(gt_inst):
    # only works for batchsize = 1 and 2d
    # using mean instead of median because its faster
    cpvs = np.zeros((2,) + gt_inst.shape, dtype=np.float32)
    ind_x, ind_y = np.where(gt_inst!=0)
    val = gt_inst[ind_x, ind_y]
    labels = np.unique(val)
    for label in labels:
        sel = val == label
        x = ind_x[sel]
        y = ind_y[sel]
        # x, y = ind_x[sel], ind_y[sel]#(gt_inst == label).long().nonzero(as_tuple=True)
        cpvs[0, x, y] = -x + x.astype(np.float32).mean()
        cpvs[1, x, y] = -y + y.astype(np.float32).mean()
    return cpvs







def get_fast_aji(true, pred):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from
    over-penalisation similar to DICE2.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.

    """
    true = np.copy(true)  # ? do we need this
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
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    return aji_score


def gen_instance_hv_maps(inst_map: np.ndarray) -> np.ndarray:
    insts = []
    for im in inst_map:
        hv = gen_instance_hv_map(im)
        cvps = inst_to_3c(im)
        insts.append(np.concatenate([im[np.newaxis, ...],hv, cvps], axis=0))


    insts = np.stack(insts)
    return insts

def get_fast_dice_2(true, pred):
    """Ensemble dice."""
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    overall_total = 0
    overall_inter = 0

    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    for true_idx in range(1, len(true_id)):
        t_mask = true_masks[true_idx]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        try:  # blinly remove background
            pred_true_overlap_id.remove(0)
        except ValueError:
            pass  # just mean no background
        for pred_idx in pred_true_overlap_id:
            p_mask = pred_masks[pred_idx]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            overall_total += total
            overall_inter += inter

    return 2 * overall_inter / overall_total



def gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
    """Obtain the horizontal and vertical distance maps for each
    nuclear instance.

    Args:
        inst_map (np.ndarray): Instance map with each instance labelled as a unique integer
            Shape: (H, W)
    Returns:
        np.ndarray: Horizontal and vertical instance map.
            Shape: (2, H, W). First dimension is horizontal (horizontal gradient (-1 to 1)),
            last is vertical (vertical gradient (-1 to 1))
    """
    orig_inst_map = inst_map.copy()  # instance ID map

    x_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(orig_inst_map))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(orig_inst_map == inst_id, np.uint8)
        inst_box = get_bounding_box(inst_map)

        # expand the box by 2px
        # Because we first pad the ann at line 207, the bboxes
        # will remain valid after expansion
        if inst_box[0] >= 2:
            inst_box[0] -= 2
        if inst_box[2] >= 2:
            inst_box[2] -= 2
        if inst_box[1] <= orig_inst_map.shape[0] - 2:
            inst_box[1] += 2
        if inst_box[3] <= orig_inst_map.shape[0] - 2:
            inst_box[3] += 2

        # improvement
        inst_map = inst_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]

        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue

        # instance center of mass, rounded to nearest pixel
        inst_com = list(center_of_mass(inst_map))

        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        ####
        x_map_box = x_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    hv_map = np.stack([(orig_inst_map>0) , x_map, y_map])
    return hv_map

if '__main__' == __name__:
    inst_pth = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/23. cellseg/train/cell_00734.npy'
    inst = np.load(inst_pth)
    hv_map = gen_instance_hv_map(inst)
    hv_map = np.transpose(hv_map, (1, 2, 0))
    hv_map = np.concatenate([(inst>0)[..., np.newaxis], hv_map], axis=2)
    det = DetectionCellPostProcessor()
    inst = det.post_process_cell_segmentation(pred_map=hv_map)

    plt.subplot(2,2,1)
    plt.imshow(hv_map[0])
    plt.subplot(2,2,2)
    plt.imshow(hv_map[1])
    plt.subplot(2,2,3)
    plt.imshow(inst)
    plt.subplot(2,2,4)
    plt.imshow(inst>0)
    plt.show()
    im_pth = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/23. cellseg/validate/cell_00076.tif'
    import tifffile
    im = tifffile.imread(im_pth)
    print(hv_map.shape)

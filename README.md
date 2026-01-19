# NucFuseRank: Dataset Fusion and Performance Ranking for Nuclei Instance Segmentation
In this paper, we evaluated publicly available H&E-stained datasets using two state-of-the-art models: CellViT and HoVerNeXt. We also introduced a fused dataset constructed from these datasets.
<img width="4292" height="2148" alt="image" src="https://github.com/user-attachments/assets/a32215de-01bc-45ba-9901-b091b357e00c" />

Ranking of datasets based on **PQ** metric.
| Dataset | Model | PQ | AJI | Dice | Precision | Recall | PQ Rank | Mean Rank |
|--------|-------|----|-----|------|-----------|--------|---------|-----------|
| PCNS | HoVerNeXt | 55.66 | 58.31 | 68.81 | 72.64 | 76.82 | 2 | 1 |
|  | CellViT | 55.14 | 57.60 | 68.20 | 73.21 | 74.67 | 1 |  |
| PUMA | HoVerNeXt | 55.75 | 57.83 | 71.57 | 78.62 | 70.83 | 1 | 2 |
|  | CellViT | 54.74 | 57.10 | 72.05 | 77.89 | 69.62 | 3 |  |
| MoNuSeg | HoVerNeXt | 53.38 | 56.08 | 68.45 | 71.05 | 73.72 | 3 | 3 |
|  | CellViT | 55.11 | 57.55 | 69.43 | 75.09 | 72.26 | 2 |  |
| CPM17 | HoVerNeXt | 51.86 | 54.49 | 65.38 | 68.05 | 71.81 | 4 | 4 |
|  | CellViT | 51.58 | 54.36 | 66.18 | 70.36 | 70.01 | 4 |  |
| TNBC | HoVerNeXt | 46.39 | 48.20 | 65.94 | 65.82 | 62.44 | 7 | 5 |
|  | CellViT | 48.67 | 52.76 | 64.85 | 65.13 | 68.85 | 5 |  |
| NuInsSeg | HoVerNeXt | 44.92 | 44.94 | 68.04 | 77.01 | 55.66 | 8 | 6 |
|  | CellViT | 48.54 | 50.38 | 66.06 | 71.77 | 63.35 | 6 |  |
| CoNSeP | HoVerNeXt | 46.41 | 50.64 | 63.44 | 61.02 | 68.24 | 6 | 7 |
|  | CellViT | 44.97 | 50.20 | 62.49 | 57.05 | 67.51 | 7 |  |
| MoNuSac | HoVerNeXt | 47.28 | 47.28 | 65.73 | 77.70 | 55.34 | 5 | 8 |
|  | CellViT | 40.40 | 40.40 | 61.39 | 71.44 | 45.84 | 9 |  |
| DSB | HoVerNeXt | 43.83 | 45.99 | 60.13 | 60.57 | 61.97 | 9 | 9 |
|  | CellViT | 38.44 | 44.24 | 56.53 | 49.26 | 59.72 | 10 |  |
| CryoNuSeg | HoVerNeXt | 35.57 | 36.57 | 60.62 | 60.31 | 48.16 | 10 | 10 |
|  | CellViT | 40.82 | 45.49 | 60.08 | 55.90 | 59.91 | 8 |  |
| **Mean** | HoVerNeXt | 48.10 | 50.03 | 65.81 | 69.27 | 64.49 |  |  |
|  | CellViT | 47.84 | 51.00 | 64.72 | 66.71 | 65.17 |  |  |

### NucFuse dataset:
NuFuse dataset is available on [figshare](https://figshare.com/)

## Acknowledgements
This project has been conducted through a joint WWTF-funded project (Grant ID: 10.47379/LS23006) between the Medical University of Vienna and Danube Private University. 
## Citation
Our paper preprint is available on arXiv: https://arxiv.org/

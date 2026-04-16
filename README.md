# Vision Models for Medical Imaging: A Hybrid Approach for PCOS Detection from Ultrasound Scans

This repository contains the code and documentation for the paper **"Vision Models for Medical Imaging: A Hybrid Approach for PCOS Detection from Ultrasound Scans"**, published in the *Journal of Physics: Conference Series* (ICSETS 2025 / IOP Publishing).

## Overview

Polycystic Ovary Syndrome (PCOS) is a common endocrine disorder affecting women of reproductive age. This project investigates whether hybrid deep learning models can improve the automatic detection of PCOS from ultrasound images. The implementation compares several pretrained CNN and Transformer architectures and introduces two ensemble-style hybrid models:

- **DenConST**: combines **DenseNet121**, **Swin Transformer**, and **ConvNeXt**
- **DenConREST**: combines **Swin Transformer**, **ConvNeXt**, **DenseNet121**, **ResNet18**, and **EfficientNetV2**

The optimized hybrid model, **DenConREST**, achieved the best performance in the study, showing strong potential as a decision-support tool for medical image analysis.

## Key Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Swin Transformer | 56.45% | 58.96% | 87.64% | 70.50% |
| ConvNeXt | 58.84% | 59.94% | 92.46% | 72.73% |
| DenseNet121 | 68.83% | 68.79% | 86.94% | 76.81% |
| ResNet18 | 59.37% | 59.37% | 100.00% | 74.50% |
| EfficientNetV2 | 79.55% | 92.79% | 71.08% | 80.50% |
| DenConST (Hybrid) | 85.69% | 88.32% | 87.47% | 87.89% |
| DenConREST (Hybrid) | **98.23%** | **97.19%** | **99.91%** | **98.49%** |

## Dataset

The experiments use a Kaggle ultrasound dataset with two classes:

- **infected** (PCOS-positive)
- **notinfected** (healthy ovaries)

The notebook performs a preprocessing step to scan the dataset recursively and copy only valid images into a clean directory before training. Invalid or corrupted images are skipped automatically.

## Methodology

The workflow in the notebook follows these steps:

1. **Image validation and cleaning** using PIL
2. **Resizing** all images to **224 × 224**
3. **Tensor conversion** and **ImageNet normalization**
4. **Training individual pretrained models**
5. **Building hybrid ensembles**
6. **Evaluating** using accuracy, precision, recall, F1 score, and confusion matrices

### Trained Models

- EfficientNetV2
- ResNet18
- DenseNet121
- Swin Transformer
- ConvNeXt

### Hybrid Models

#### DenConST
A three-model ensemble averaging predictions from:
- DenseNet121
- Swin Transformer
- ConvNeXt

#### DenConREST
A five-model ensemble averaging predictions from:
- Swin Transformer
- ConvNeXt
- DenseNet121
- ResNet18
- EfficientNetV2

## Implementation Notes

- Framework: **PyTorch**
- Vision library: **torchvision**
- Transformer / model zoo: **timm**
- Metrics: **scikit-learn**
- Visualization: **matplotlib**, **seaborn**, **plotly**
- Training environment: **Kaggle GPU notebook**

The notebook includes reusable training and evaluation functions, confusion matrix plotting, and a final interactive visualization for the best ensemble model.

## Repository Structure

```text
.
├── pcos-detection.ipynb
├── README.md
└── data/
    ├── train/
    │   ├── infected/
    │   └── notinfected/
    └── test/
        ├── infected/
        └── notinfected/
```

## Requirements

Install the following Python packages:

```bash
pip install torch torchvision timm scikit-learn matplotlib seaborn plotly pandas pillow tqdm
```

## Running the Notebook

1. Download or prepare the dataset in the expected `train/` and `test/` directory format.
2. Update the dataset paths in the notebook if needed.
3. Run the preprocessing cell to clean corrupted images.
4. Train the individual models.
5. Run the ensemble evaluation cells.
6. Review the classification reports and confusion matrices.

## Citation

If you use this work, please cite the paper as:

```bibtex
@inproceedings{mahmudul2026vision,
  title={Vision Models for Medical Imaging: A Hybrid Approach for PCOS Detection from Ultrasound Scans},
  author={Mahmudul Hoque, Md and Mehedi Hassain, Md and Rahaman, Muntakimur and Towhidul Islam, Md and Rani, Shaista and Sharif Mollah, Md},
  booktitle={Journal of Physics: Conference Series},
  volume={3191},
  number={1},
  pages={012120},
  year={2026},
  organization={IOP Publishing}
}
```

## Paper Information

- **Conference**: International Conference on Systems Engineering, Technology and Sustainable Solutions (ICSETS 2025)
- **Journal**: Journal of Physics: Conference Series
- **Publisher**: IOP Publishing
- **DOI**: 10.1088/1742-6596/3191/1/012120

## Authors

Md Mahmudul Hoque, Md Mehedi Hassain, Muntakimur Rahaman, Md. Towhidul Islam, Shaista Rani, and Md Sharif Mollah

## Acknowledgment

This project explores AI-driven medical imaging for PCOS detection and demonstrates how hybrid CNN-Transformer architectures can improve diagnostic performance in biomedical applications.

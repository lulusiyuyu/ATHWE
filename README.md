# ATHWE: Adaptive Temporal Expert Routing with Hierarchical Wavelet Enhancement for Multi-Modal Sequential Recommendation

## Introduction

ATHWE is a multi-modal sequential recommendation model that combines adaptive temporal expert routing with hierarchical wavelet enhancement. The model processes multiple modalities (item IDs, text, and images) through:

- **Adaptive Temporal Expert Routing**: Dynamically routes features through specialized experts based on temporal patterns
- **Hierarchical Wavelet Enhancement**: Uses discrete wavelet transform (DWT) for multi-scale feature fusion
- **Contrastive Learning**: Employs multiple contrastive learning objectives
- **Transformer Encoders**: Processes sequential patterns for each modality

## Requirements

```bash
pip install torch
pip install recbole
pip install pytorch-wavelets
pip install scikit-learn-extra
pip install numpy
```

## Dataset Preparation

Place your dataset in the `dataset/` folder with the following structure:
```
dataset/
└── [DATASET_NAME]/
    ├── [DATASET_NAME].inter    # User-item interaction data
    ├── txt_emb.pt              # Pre-extracted text embeddings (768-dim)
    ├── img_emb.pt              # Pre-extracted image embeddings (768-dim)
    └── cat.pt                  # Category attributes
```

Create a corresponding configuration file in `config/[DATASET_NAME].yaml`.

## Usage

Run the following command to train the model:

```bash
python run_athwe.py
```

Modify the dataset name in `run_athwe.py`:

```python
dataset = 'Home_and_Kitchen'  # Change to your dataset name
run_recbole(model='ATHWE', dataset=dataset,
            config_file_list=['./config/data.yaml', f'./config/{dataset}.yaml'])
```

## Acknowledgments

Our implementation is based on [RecBole](https://github.com/RUCAIBox/RecBole) and [UniSRec](https://github.com/RUCAIBox/UniSRec). This work is also built upon [HM4SR](https://github.com/SStarCCat/HM4SR) for both implementation and dataset preparation. Thanks for the splendid codes from these authors.

## Citation

If you find this code useful in your research, please consider citing:

```bibtex

```

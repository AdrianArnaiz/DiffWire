# DiffWire: Inductive Graph Rewiring via the Lovász Bound

**Accepted at the First Learning on Graphs Conference 2022**

[![LoG](https://img.shields.io/badge/Published%20-Learning%20on%20Graphs-blue.svg)](https://openreview.net/forum?id=IXvfIex0mX6f&noteId=t5zJZuEIy1y)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffwire-inductive-graph-rewiring-via-the/graph-classification-on-imdb-binary)](https://paperswithcode.com/sota/graph-classification-on-imdb-binary?p=diffwire-inductive-graph-rewiring-via-the)



<img width="942" alt="image" src="https://user-images.githubusercontent.com/60975511/169371484-f31a1caa-0249-4c22-aba4-055cda206241.png">

$$
\left| \frac{1}{vol(G)}CT_{uv}-\left(\frac{1}{d_u} + \frac{1}{d_v}\right)\right|\le \frac{1}{\lambda_2}\frac{2}{d_{min}}
$$

## Dependencies

Conda environment
```
conda create --name <env> --file requirements.txt
```

or

```
conda env create -f environment_experiments.yml
conda activate DiffWire
```
## Code organization

* `datasets/`: script for creating synthetic datasets. For non-synthetic ones: we use PyG in `train.py`
* `layers/`: Implementation of the proposed **GAP-Layer**, **CT-Layer**, and the baseline MinCutPool (based on his repo).
* `tranforms/`: Implementation og graph preprocessing baselines DIGL and SDRF, both based on the official repositories of the work.
* `trained_models/`: files with the weight of some trained models.
* `nets.py`: Implementation of GNNs used in our experiments.
* `train.py`: Script with inline arguments for running the experiments.

## Run experiments
```python
python train.py --dataset REDDIT-BINARY --model CTNet --cuda cuda:0
python train.py --dataset REDDIT-BINARY --model GAPNet --derivative laplacian --cuda cuda:0
python train.py --dataset REDDIT-BINARY --model GAPNet --derivative normalizeed --cuda cuda:0
```

`experiments_all.sh` list all the experiments.

## Citation

```
@inproceedings{
arnaiz-rodriguez2022diffwire,
title={DiffWire: Inductive Graph Rewiring via the Lov\'asz Bound},
author={Adrian Arnaiz-Rodriguez and Ahmed Begga and Francisco Escolano and Nuria Oliver},
booktitle={Learning on Graphs Conference},
year={2022},
url={https://openreview.net/forum?id=IXvfIex0mX6f}
}
```
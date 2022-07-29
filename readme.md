# Inductive Graph Rewiring via the Lovász Bound
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

**Under review**
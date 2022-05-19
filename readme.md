# Inductive Graph Rewiring via the Lovász Bound
<img width="950" alt="image" src="https://user-images.githubusercontent.com/60975511/169368951-589a5890-099c-4f38-b189-b9ef5abd3b4b.png">


## Dependencies

Conda environment
```
conda create --name <env> --file requirements.txt
```

For pip environment, export from the previous installed conda environment
```
conda activate <env>
conda install pip
pip freeze > requirements.txt
```
If it doesn´t work, try:
```
conda env create -f environment_experiments.yml
conda activate DiffWire
```
## Code organization


## Run experiments
```python
python train.py --dataset REDDIT-BINARY --model CTNet --cuda cuda:0
python train.py --dataset REDDIT-BINARY --model GAPNet --derivative laplacian --cuda cuda:0
```

## Citation


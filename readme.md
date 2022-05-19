# Inductive Graph Rewiring via the Lovász Bound
<img width="955" alt="image" src="https://user-images.githubusercontent.com/60975511/169361386-b08f83e9-c302-4ebe-8b1c-dea70fa709b1.png">


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
```
## Code organization


## Run experiments
```python
python train.py --dataset REDDIT-BINARY --model CTNet --cuda cuda:0
python train.py --dataset REDDIT-BINARY --model GAPNet --derivative laplacian --cuda cuda:0
```

## Citation


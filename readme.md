# Inductive Graph Rewiring via the Lovász Bound
<img width="959" alt="image" src="https://user-images.githubusercontent.com/60975511/169364961-50279da9-728c-4188-afde-d21fd5ef4d10.png">


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


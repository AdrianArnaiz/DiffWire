python train.py --dataset IMDB-BINARY --model CTNet
python train.py --dataset IMDB-BINARY --model GAPNet --derivative laplacian
python train.py --dataset IMDB-BINARY --model GAPNet --derivative normalized

python train.py --dataset REDDIT-BINARY --model CTNet
python train.py --dataset REDDIT-BINARY --model GAPNet --derivative laplacian
python train.py --dataset REDDIT-BINARY --model GAPNet --derivative normalized

python train.py --dataset COLLAB --model CTNet
python train.py --dataset COLLAB --model GAPNet --derivative laplacian
python train.py --dataset COLLAB --model GAPNet --derivative normalized

python train.py --dataset ENZYMES --model CTNet
python train.py --dataset ENZYMES --model GAPNet --derivative laplacian
python train.py --dataset ENZYMES --model GAPNet --derivative normalized

python train.py --dataset PROTEINS --model CTNet
python train.py --dataset PROTEINS --model GAPNet --derivative laplacian
python train.py --dataset PROTEINS --model GAPNet --derivative normalized

python train.py --dataset MUTAG --model CTNet
python train.py --dataset MUTAG --model GAPNet --derivative laplacian
python train.py --dataset MUTAG --model GAPNet --derivative normalized
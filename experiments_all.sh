python train.py --dataset MNIST --model MinCutNet --cuda cuda:0
python train.py --dataset MNIST --model CTNet --cuda cuda:0
python train.py --dataset MNIST --model GAPNet --derivative laplacian --cuda cuda:0
python train.py --dataset MNIST --model GAPNet --derivative normalized --cuda cuda:0
python train.py --dataset MNIST --model GAPNet --derivative normalizedv2 --cuda cuda:0

python train.py --dataset CIFAR10 --model MinCutNet --cuda cuda:0
python train.py --dataset CIFAR10 --model CTNet --cuda cuda:0
python train.py --dataset CIFAR10 --model GAPNet --derivative laplacian --cuda cuda:0
python train.py --dataset CIFAR10 --model GAPNet --derivative normalized --cuda cuda:0
python train.py --dataset CIFAR10 --model GAPNet --derivative normalizedv2 --cuda cuda:0

python train.py --dataset REDDIT-BINARY --model MinCutNet --cuda cuda:0
python train.py --dataset REDDIT-BINARY --model CTNet --cuda cuda:0
python train.py --dataset REDDIT-BINARY --model GAPNet --derivative laplacian --cuda cuda:0
python train.py --dataset REDDIT-BINARY --model GAPNet --derivative normalized --cuda cuda:0
python train.py --dataset REDDIT-BINARY --model GAPNet --derivative normalizedv2 --cuda cuda:0

python train.py --dataset IMDB-BINARY --model MinCutNet --cuda cuda:0
python train.py --dataset IMDB-BINARY --model CTNet --cuda cuda:0
python train.py --dataset IMDB-BINARY --model GAPNet --derivative laplacian --cuda cuda:0
python train.py --dataset IMDB-BINARY --model GAPNet --derivative normalized --cuda cuda:0
python train.py --dataset IMDB-BINARY --model GAPNet --derivative normalizedv2 --cuda cuda:0

python train.py --dataset COLLAB --model MinCutNet --cuda cuda:0
python train.py --dataset COLLAB --model CTNet --cuda cuda:0
python train.py --dataset COLLAB --model GAPNet --derivative laplacian --cuda cuda:0
python train.py --dataset COLLAB --model GAPNet --derivative normalized --cuda cuda:0
python train.py --dataset COLLAB --model GAPNet --derivative normalizedv2 --cuda cuda:0

python train.py --dataset ENZYMES --model MinCutNet --cuda cuda:0
python train.py --dataset ENZYMES --model CTNet --cuda cuda:0
python train.py --dataset ENZYMES --model GAPNet --derivative laplacian --cuda cuda:0
python train.py --dataset ENZYMES --model GAPNet --derivative normalized --cuda cuda:0
python train.py --dataset ENZYMES --model GAPNet --derivative normalizedv2 --cuda cuda:0

python train.py --dataset PROTEINS --model MinCutNet --cuda cuda:0
python train.py --dataset PROTEINS --model CTNet --cuda cuda:0
python train.py --dataset PROTEINS --model GAPNet --derivative laplacian --cuda cuda:0
python train.py --dataset PROTEINS --model GAPNet --derivative normalized --cuda cuda:0
python train.py --dataset PROTEINS --model GAPNet --derivative normalizedv2 --cuda cuda:0

python train.py --dataset MUTAG --model MinCutNet --cuda cuda:0
python train.py --dataset MUTAG --model CTNet --cuda cuda:0
python train.py --dataset MUTAG --model GAPNet --derivative laplacian --cuda cuda:0
python train.py --dataset MUTAG --model GAPNet --derivative normalized --cuda cuda:0
python train.py --dataset MUTAG --model GAPNet --derivative normalizedv2 --cuda cuda:0
python train.py --dataset MNIST --model CTNet --cuda cuda:0
python train.py --dataset MNIST --model GAPNet --derivative laplacian --cuda cuda:0
python train.py --dataset MNIST --model GAPNet --derivative normalized --cuda cuda:0
python train.py --dataset MNIST --model GAPNet --derivative normalizedv2 --cuda cuda:0

python train.py --dataset CIFAR10 --model CTNet --cuda cuda:0
python train.py --dataset CIFAR10 --model GAPNet --derivative laplacian --cuda cuda:0
python train.py --dataset CIFAR10 --model GAPNet --derivative normalized --cuda cuda:0
python train.py --dataset CIFAR10 --model GAPNet --derivative normalizedv2 --cuda cuda:0
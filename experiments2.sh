python train.py --dataset MNIST --model CTNet
python train.py --dataset MNIST --model GAPNet --derivative laplacian
python train.py --dataset MNIST --model GAPNet --derivative normalized

python train.py --dataset CIFAR10 --model CTNet
python train.py --dataset CIFAR10 --model GAPNet --derivative laplacian
python train.py --dataset CIFAR10 --model GAPNet --derivative normalized
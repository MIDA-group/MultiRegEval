# VoxelMorph
mkdir ./logs

# Initial test
## unsupervised
nohup python train_affine.py --data_root ../Datasets/Eliceiri_patches/patch_trans10_rot5 --model_name unsupervised --epochs 1500 --steps_per_epoch 50 --batch_size 8 --gpu 1 > ./logs/out_eliceiri_unsupervised.file 2>&1 &

## supervised
nohup python train_affine.py --data_root ../Datasets/Eliceiri_patches/patch_trans10_rot5 --supervised --model_name supervised --epochs 1500 --steps_per_epoch 50 --batch_size 8 --gpu 0 > ./logs/out_eliceiri_supervised.file 2>&1 &


# BF2SHG, bidir loss
## unsupervised
nohup python train_affine.py --data_root ../Datasets/Eliceiri_patches/patch_trans0-20_rot0-5 --a2b 0 --model_name unsupervised_rot0-5_b2a --epochs 500 --steps_per_epoch 50 --batch_size 8 --gpu 0 > ./logs/out_eliceiri_unsupervised_rot0-5_b2a.file 2>&1 &
nohup python train_affine.py --data_root ../Datasets/Eliceiri_patches/patch_trans20-40_rot5-10 --a2b 0 --model_name unsupervised_rot5-10_b2a --epochs 500 --steps_per_epoch 50 --batch_size 8 --gpu 0 > ./logs/out_eliceiri_unsupervised_rot5-10_b2a.file 2>&1 &
nohup python train_affine.py --data_root ../Datasets/Eliceiri_patches/patch_trans40-60_rot10-15 --a2b 0 --model_name unsupervised_rot10-15_b2a --epochs 500 --steps_per_epoch 50 --batch_size 8 --gpu 0 > ./logs/out_eliceiri_unsupervised_rot10-15_b2a.file 2>&1 &
nohup python train_affine.py --data_root ../Datasets/Eliceiri_patches/patch_trans60-80_rot15-20 --a2b 0 --model_name unsupervised_rot15-20_b2a --epochs 500 --steps_per_epoch 50 --batch_size 8 --gpu 0 > ./logs/out_eliceiri_unsupervised_rot15-20_b2a.file 2>&1 &

## supervised
nohup python train_affine.py --data_root ../Datasets/Eliceiri_patches/patch_trans0-20_rot0-5 --a2b 0 --supervised --model_name supervised_rot0-5_b2a --epochs 500 --steps_per_epoch 50 --batch_size 8 --gpu 2 > ./logs/out_eliceiri_supervised_rot0-5_b2a.file 2>&1 &
nohup python train_affine.py --data_root ../Datasets/Eliceiri_patches/patch_trans20-40_rot5-10 --a2b 0 --supervised --model_name supervised_rot5-10_b2a --epochs 500 --steps_per_epoch 50 --batch_size 8 --gpu 2 > ./logs/out_eliceiri_supervised_rot5-10_b2a.file 2>&1 &
nohup python train_affine.py --data_root ../Datasets/Eliceiri_patches/patch_trans40-60_rot10-15 --a2b 0 --supervised --model_name supervised_rot10-15_b2a --epochs 500 --steps_per_epoch 50 --batch_size 8 --gpu 2 > ./logs/out_eliceiri_supervised_rot10-15_b2a.file 2>&1 &
nohup python train_affine.py --data_root ../Datasets/Eliceiri_patches/patch_trans60-80_rot15-20 --a2b 0 --supervised --model_name supervised_rot15-20_b2a --epochs 500 --steps_per_epoch 50 --batch_size 8 --gpu 2 > ./logs/out_eliceiri_supervised_rot15-20_b2a.file 2>&1 &

## train all
mkdir ./logs
nohup ./train_unsupervised.sh &
nohup ./train_supervised.sh &

## train MNIST
nohup python mnist_rigid.py --a2b 0 --model_name mnist_us --epochs 500 --batch_size 256 --gpu 0 > ./logs/out_mnist_us_b2a.file 2>&1 &
nohup python mnist_rigid.py --a2b 0 --supervised --model_name mnist_su --epochs 500 --batch_size 256 --gpu 1 > ./logs/out_mnist_su_b2a.file 2>&1 &


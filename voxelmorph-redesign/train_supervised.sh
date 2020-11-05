# VoxelMorph

# BF2SHG, bidir loss
## supervised
python train_affine.py --data_root ../Datasets/Eliceiri_patches/patch_trans0-20_rot0-5 --a2b 0 --supervised --model_name supervised_rot0-5_b2a --epochs 500 --steps_per_epoch 50 --batch_size 8 --gpu 2 > ./logs/eliceiri_supervised_rot0-5_b2a.out 2>&1
python train_affine.py --data_root ../Datasets/Eliceiri_patches/patch_trans20-40_rot5-10 --a2b 0 --supervised --model_name supervised_rot5-10_b2a --epochs 500 --steps_per_epoch 50 --batch_size 8 --gpu 2 > ./logs/eliceiri_supervised_rot5-10_b2a.out 2>&1
python train_affine.py --data_root ../Datasets/Eliceiri_patches/patch_trans40-60_rot10-15 --a2b 0 --supervised --model_name supervised_rot10-15_b2a --epochs 500 --steps_per_epoch 50 --batch_size 8 --gpu 2 > ./logs/eliceiri_supervised_rot10-15_b2a.out 2>&1
python train_affine.py --data_root ../Datasets/Eliceiri_patches/patch_trans60-80_rot15-20 --a2b 0 --supervised --model_name supervised_rot15-20_b2a --epochs 500 --steps_per_epoch 50 --batch_size 8 --gpu 2 > ./logs/eliceiri_supervised_rot15-20_b2a.out 2>&1

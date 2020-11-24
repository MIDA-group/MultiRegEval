# Balvan's data

## Train
nohup python train.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/balvan_cyc_train/fold1 --batch_size 64 --crop_size 256 --name balvan_drit_train_fold1 --concat 0 --n_ep 1200 --input_dim_a 1 --input_dim_b 1 --gpu 1 > ./outfiles/balvan_fold1.file 2>&1 &
wait


## Test
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/balvan_cyc_train/fold1/ --resize_size 600 --crop_size 600 --a2b 1 --name balvan_drit_train_fold1_a2b --concat 0 --resume ../results/balvan_drit_train_fold1/01199.pth --input_dim_a 1 --input_dim_b 1 
# strangely, --gpu can only be 0
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/balvan_cyc_train/fold1/ --resize_size 600 --crop_size 600 --a2b 0 --name balvan_drit_train_fold1_b2a --concat 0 --resume ../results/balvan_drit_train_fold1/01199.pth --input_dim_a 1 --input_dim_b 1 
# 以上命令已运行

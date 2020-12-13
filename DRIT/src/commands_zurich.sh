# Zurich's data

## Train
nohup python train.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/zurich_cyc_train/fold1 --batch_size 64 --crop_size 256 --name zurich_drit_train_fold1 --concat 1 --n_ep 1200 --gpu 0 > ./outfiles/zurich_fold1.file 2>&1 &
# 以上命令已运行


## Test
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/zurich_cyc_train/fold1/ --resize_size 600 --crop_size 600 --a2b 1 --name zurich_drit_train_fold1_a2b --concat 1 --resume ../results/zurich_drit_train_fold1/01199.pth
# strangely, --gpu can only be 0
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/zurich_cyc_train/fold1/ --resize_size 600 --crop_size 600 --a2b 0 --name zurich_drit_train_fold1_b2a --concat 1 --resume ../results/zurich_drit_train_fold1/01199.pth

nohup python train.py --dataroot ../datasets/MPMSkin/ --batch_size 32 --resize_size 256 --crop_size 256 --name MPMSkin --concat 0 --n_ep 1200 --gpu 1 > out.file 2>&1 &

python test_transfer.py --dataroot ../datasets/MPMSkin/ --resize_size 256 --crop_size 256 --a2b 1 --name MPMSkin_a2b_guided --concat 0 --resume ../results/MPMSkin/01199.pth
python test_transfer.py --dataroot ../datasets/MPMSkin/ --resize_size 256 --crop_size 256 --a2b 0 --name MPMSkin_b2a_guided --concat 0 --resume ../results/MPMSkin/01199.pth
python test.py --dataroot ../datasets/MPMSkin/ --resize_size 256 --crop_size 256 --a2b 1 --name MPMSkin_a2b_random --concat 0 --resume ../results/MPMSkin/01199.pth
python test.py --dataroot ../datasets/MPMSkin/ --resize_size 256 --crop_size 256 --a2b 0 --name MPMSkin_b2a_random --concat 0 --resume ../results/MPMSkin/01199.pth



# Eliceiri's data

## Train
nohup python train.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_cyc_train --batch_size 32 --resize_size 256 --crop_size 256 --name eliceiri_drit_train --concat 0 --n_ep 1200 --gpu 1 > out_eliceiri.file 2>&1 &

## Test
### rotation
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_cyc_test/rotation/ --resize_size 834 --crop_size 834 --a2b 1 --name eliceiri_drit_rotation_a2b --concat 0 --resume ../results/eliceiri_drit_train/01199.pth
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_cyc_test/rotation/ --resize_size 834 --crop_size 834 --a2b 0 --name eliceiri_drit_rotation_b2a --concat 0 --resume ../results/eliceiri_drit_train/01199.pth
### processed
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_cyc_test/processed/ --resize_size 834 --crop_size 834 --a2b 1 --name eliceiri_drit_processed_a2b --concat 0 --resume ../results/eliceiri_drit_train/01199.pth
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_cyc_test/processed/ --resize_size 834 --crop_size 834 --a2b 0 --name eliceiri_drit_processed_b2a --concat 0 --resume ../results/eliceiri_drit_train/01199.pth



# ## Train with inverted data A
# nohup python train.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_cyc_train_Ainverted --batch_size 32 --resize_size 256 --crop_size 256 --name eliceiri_drit_train_Ainverted --concat 0 --n_ep 1200 --gpu 1 > out_eliceiri_Ainverted.file 2>&1 &
# python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_cyc_test/rotation/ --resize_size 834 --crop_size 834 --a2b 1 --name eliceiri_drit_Ainverted_rotation_a2b --concat 0 --resume ../results/eliceiri_drit_train_Ainverted/01199.pth
# python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_cyc_test/rotation/ --resize_size 834 --crop_size 834 --a2b 0 --name eliceiri_drit_Ainverted_rotation_b2a --concat 0 --resume ../results/eliceiri_drit_train_Ainverted/01199.pth 




# Balvan's data

## Train
nohup python train.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/balvan_cyc_train256/fold1 --batch_size 32 --resize_size 256 --crop_size 256 --name balvan_drit_train_fold1 --concat 0 --n_ep 1200 --input_dim_a 1 --input_dim_b 1 --gpu 1 > out_balvan.file 2>&1 &


## Test
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/balvan_cyc_train256/fold1/ --resize_size 600 --crop_size 600 --a2b 1 --name balvan_drit_train_fold1_a2b --concat 0 --resume ../results/balvan_drit_train_fold1/01199.pth --input_dim_a 1 --input_dim_b 1 
# strangely, --gpu can only be 0
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/balvan_cyc_train256/fold1/ --resize_size 600 --crop_size 600 --a2b 0 --name balvan_drit_train_fold1_b2a --concat 0 --resume ../results/balvan_drit_train_fold1/01199.pth --input_dim_a 1 --input_dim_b 1 
# 以上命令已运行

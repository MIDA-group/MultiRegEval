# Run: nohup ./commands_balvan.sh {fold} {gpu_id} &

# For Balvan's data 
# ### extract Balvan modalities
# python ./utils/extract_balvan_modalites.py

# ### train/test split
# python ./utils/prepare_Balvan.py

# ## prepare training data
# cd pytorch-CycleGAN-and-pix2pix/


python datasets/combine_A_and_B.py --fold_A ../Datasets/Balvan_temp/fold$1/A --fold_B ../Datasets/Balvan_temp/fold$1/B --fold_AB ./datasets/balvan_train/fold$1

## Train
nohup python train.py --display_id -1 --dataroot ./datasets/balvan_train/fold$1 --name balvan_p2p_train_fold$1_a2b --model pix2pix --direction AtoB --input_nc 1 --output_nc 1 --n_epochs 100 --n_epochs_decay 100 --preprocess mix --load_size 256 --crop_size 256 --batch_size 64 --gpu_ids $2 > out_a2b.file 2>&1 &
# Training log at ./checkpoints/balvan_p2p_train_fold$1_a2b/loss_log.txt
wait
nohup python train.py --display_id -1 --dataroot ./datasets/balvan_train/fold$1 --name balvan_p2p_train_fold$1_b2a --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --n_epochs 100 --n_epochs_decay 100 --preprocess mix --load_size 256 --crop_size 256 --batch_size 64 --gpu_ids $2 > out_b2a.file 2>&1 &
wait

# ## Test
# python test.py --dataroot ./datasets/balvan_train/fold$1 --name balvan_p2p_train_fold$1_a2b --model pix2pix --num_test 99999 --direction AtoB --input_nc 1 --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids $2
# python test.py --dataroot ./datasets/balvan_train/fold$1 --name balvan_p2p_train_fold$1_b2a --model pix2pix --num_test 99999 --direction BtoA --input_nc 1 --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids $2

# ### unpad results
# python ../utils/unpad_results.py -p ./results/balvan_p2p_train_fold$1_a2b/test_latest/images --width 600 --height 600
# python ../utils/unpad_results.py -p ./results/balvan_p2p_train_fold$1_b2a/test_latest/images --width 600 --height 600

# 以上命令已运行


# cycleGAN

## Balvan's data
### prepare training data
mkdir -p datasets/balvan_cyc_train/fold$1
cp -r ../Datasets/Balvan_temp/fold$1/A/train/ ./datasets/balvan_cyc_train/fold$1/trainA
cp -r ../Datasets/Balvan_temp/fold$1/B/train/ ./datasets/balvan_cyc_train/fold$1/trainB
### prepare test data
cp -r ../Datasets/Balvan_temp/fold$1/A/test/ ./datasets/balvan_cyc_train/fold$1/testA
cp -r ../Datasets/Balvan_temp/fold$1/B/test/ ./datasets/balvan_cyc_train/fold$1/testB
# 以上命令已运行
### Train
wait
# here input&output sizes are fixed to 1
nohup python train.py --display_id -1 --dataroot ./datasets/balvan_cyc_train/fold$1 --name balvan_cyc_train_fold$1 --model cycle_gan --preprocess mix --load_size 256 --crop_size 256 --batch_size 4 --input_nc 1 --output_nc 1 --gpu_ids $2 > out.file 2>&1 &
# batch_size can only be 1 if using single GPU
# Training log at ./checkpoints/balvan_cyc_train_fold$1/loss_log.txt

# wait
# ### Test
# # here input&output sizes are fixed to 1
# python test.py --dataroot ./datasets/balvan_cyc_train/fold$1/ --name balvan_cyc_train_fold$1 --model cycle_gan --num_test 99999 --batch_size 4 --preprocess pad --divisor 256 --input_nc 1 --output_nc 1 --gpu_ids $2

# ### unpad results
# python ../utils/unpad_results.py -p ./results/balvan_cyc_train_fold$1/test_latest/images --width 600 --height 600

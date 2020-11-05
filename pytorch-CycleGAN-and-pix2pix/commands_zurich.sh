# Run: nohup ./commands_zurich.sh {fold} {gpu_id} &

# For Zurich's data 
# ### extract Zurich modalities
# python ./utils/extract_zurich_modalites.py

# ### train/test split
# python ./utils/prepare_Zurich.py

# ## prepare training data
# cd pytorch-CycleGAN-and-pix2pix/
# # python ../utils/prepare_Zurich.py

python datasets/combine_A_and_B.py --fold_A ../Datasets/Zurich_temp/fold$1/A --fold_B ../Datasets/Zurich_temp/fold$1/B --fold_AB ./datasets/zurich_train/fold$1

## Train
nohup python train.py --display_id -1 --dataroot ./datasets/zurich_train/fold$1 --name zurich_p2p_train_fold$1_a2b --model pix2pix --direction AtoB --input_nc 1 --output_nc 3 --n_epochs 100 --n_epochs_decay 100 --serial_batches --load_size 256 --crop_size 256 --batch_size 64 --gpu_ids $2 > out_a2b.file 2>&1 &
wait
# Training log at ./checkpoints/zurich_p2p_train_fold$1_a2b/loss_log.txt
nohup python train.py --display_id -1 --dataroot ./datasets/zurich_train/fold$1 --name zurich_p2p_train_fold$1_b2a --model pix2pix --direction BtoA --input_nc 3 --output_nc 1 --n_epochs 100 --n_epochs_decay 100 --serial_batches --load_size 256 --crop_size 256 --batch_size 64 --gpu_ids $2 > out_b2a.file 2>&1 &
wait
# 以上命令已运行


# ## Test
# python test.py --dataroot ./datasets/zurich_train/fold$1 --name zurich_p2p_train_fold$1_a2b --model pix2pix --num_test 99999 --direction AtoB --input_nc 1 --output_nc 3 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids $2
# python test.py --dataroot ./datasets/zurich_train/fold$1 --name zurich_p2p_train_fold$1_b2a --model pix2pix --num_test 99999 --direction BtoA --input_nc 3 --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids $2

### unpad results
# python ../utils/unpad_results.py -p ./results/zurich_p2p_train_fold$1_a2b/test_latest/images --width 600 --height 600
# python ../utils/unpad_results.py -p ./results/zurich_p2p_train_fold$1_b2a/test_latest/images --width 600 --height 600



# cycleGAN

## Zurich's data

### prepare training data
mkdir -p datasets/zurich_cyc_train/fold$1
cp -r ../Datasets/Zurich_temp/fold$1/A/train/ ./datasets/zurich_cyc_train/fold$1/trainA
cp -r ../Datasets/Zurich_temp/fold$1/B/train/ ./datasets/zurich_cyc_train/fold$1/trainB
### prepare test data
cp -r ../Datasets/Zurich_temp/fold$1/A/test/ ./datasets/zurich_cyc_train/fold$1/testA
cp -r ../Datasets/Zurich_temp/fold$1/B/test/ ./datasets/zurich_cyc_train/fold$1/testB
### Train
wait
nohup python train.py --display_id -1 --dataroot ./datasets/zurich_cyc_train/fold$1 --name zurich_cyc_train_fold$1 --model cycle_gan --serial_batches --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids $2 > out.file 2>&1 &
# batch_size can only be 1 if using single GPU
# Training log at ./checkpoints/zurich_cyc_train_fold$1/loss_log.txt
# 以上命令已运行

# wait

# ### Test
# python test.py --dataroot ./datasets/zurich_cyc_train/fold$1/ --name zurich_cyc_train_fold$1 --model cycle_gan --num_test 99999 --batch_size 1 --preprocess pad --divisor 256 --gpu_ids $2

### unpad results
# python ../utils/unpad_results.py -p ./results/zurich_cyc_train_fold$1/test_latest/images --width 600 --height 600

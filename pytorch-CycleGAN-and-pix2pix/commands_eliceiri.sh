# Run: nohup ./commands_eliceiri.sh {gpu_id} &

# For Eliceiri's data 
# ### extract Eliceiri modalities
# python ./utils/extract_eliceiri_modalites.py

# ### train/test split
# python ./utils/prepare_Eliceiri.py

# ## prepare training data
# cd pytorch-CycleGAN-and-pix2pix/
# # python ../utils/prepare_Eliceiri.py

# 以上命令已运行
python datasets/combine_A_and_B.py --fold_A ../Datasets/Eliceiri_temp/A --fold_B ../Datasets/Eliceiri_temp/B --fold_AB ./datasets/eliceiri_train


## Train
nohup python train.py --display_id -1 --dataroot ./datasets/eliceiri_train --name eliceiri_p2p_train_a2b --model pix2pix --direction AtoB --input_nc 1 --output_nc 3 --n_epochs 100 --n_epochs_decay 100 --preprocess mix --crop_size 256 --batch_size 64 --gpu_ids 1 > out_a2b.file 2>&1 &
wait
# Training log at ./checkpoints/eliceiri_p2p_train_a2b/loss_log.txt
nohup python train.py --display_id -1 --dataroot ./datasets/eliceiri_train --name eliceiri_p2p_train_b2a --model pix2pix --direction BtoA --input_nc 3 --output_nc 1 --n_epochs 100 --n_epochs_decay 100 --preprocess mix --crop_size 256 --batch_size 64 --gpu_ids 1 > out_b2a.file 2>&1 &
wait

# ## Test
# python test.py --dataroot ./datasets/eliceiri_train --name eliceiri_p2p_train_a2b --model pix2pix --num_test 99999 --direction AtoB --input_nc 1 --output_nc 3 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 1
# python test.py --dataroot ./datasets/eliceiri_train --name eliceiri_p2p_train_b2a --model pix2pix --num_test 99999 --direction BtoA --input_nc 3 --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 1

# ### unpad results
# python ../utils/unpad_results.py -p ./results/eliceiri_p2p_train_a2b/test_latest/images --width 834 --height 834
# python ../utils/unpad_results.py -p ./results/eliceiri_p2p_train_b2a/test_latest/images --width 834 --height 834



# cycleGAN

## Eliceiri's data

#### image width 256 re-training
### prepare training data
mkdir -p datasets/eliceiri_cyc_train
cp -r ../Datasets/Eliceiri_temp/A/train/ ./datasets/eliceiri_cyc_train/trainA
cp -r ../Datasets/Eliceiri_temp/B/train/ ./datasets/eliceiri_cyc_train/trainB
### prepare test data
cp -r ../Datasets/Eliceiri_temp/A/test/ ./datasets/eliceiri_cyc_train/testA
cp -r ../Datasets/Eliceiri_temp/B/test/ ./datasets/eliceiri_cyc_train/testB
### Train
wait
# here input&output sizes are fixed to 1
nohup python train.py --display_id -1 --dataroot ./datasets/eliceiri_cyc_train --name eliceiri_cyc_train --model cycle_gan --preprocess mix --crop_size 256 --batch_size 4 --gpu_ids 1 > out.file 2>&1 &
# batch_size can only be 1 if using single GPU
# Training log at ./checkpoints/eliceiri_cyc_train/loss_log.txt

# wait

# ### Test
# # here input&output sizes are fixed to 1
# python test.py --dataroot ./datasets/eliceiri_cyc_train/ --name eliceiri_cyc_train --model cycle_gan --num_test 99999 --batch_size 4 --preprocess pad --divisor 256 --gpu_ids 1

# ### unpad results
# python ../utils/unpad_results.py -p ./results/eliceiri_cyc_train/test_latest/images --width 834 --height 834
# # 以上命令已运行

# Run: ./commands_eliceiri.sh {gpu_id}


## Eliceiri's data
### prepare training data
mkdir -p data/eliceiri/train

### MUST RUN ON MIDA1!!! ###
ln -s /data2/jiahao/Registration/Datasets/Eliceiri_temp/A/train data/eliceiri/train/A
ln -s /data2/jiahao/Registration/Datasets/Eliceiri_temp/B/train data/eliceiri/train/B
ln -s /data2/jiahao/Registration/stargan-v2/data/eliceiri/train data/eliceiri/val
### MUST RUN ON MIDA1!!! ###

# cp -r ../Datasets/Eliceiri_temp/A/train ./data/eliceiri/train/A
# cp -r ../Datasets/Eliceiri_temp/B/train ./data/eliceiri/train/B
# use train as validation set
# cp -r data/eliceiri/train data/eliceiri/val


### Train
wait
# here input&output sizes are fixed to 1
mkdir -p ./checkpoints/eliceiri_train
# 以上命令已运行
CUDA_VISIBLE_DEVICES=$1 nohup python main.py --mode train --img_size 256 --num_domains 2 --w_hpf 0 \
	--train_img_dir data/eliceiri/train \
	--val_img_dir data/eliceiri/val \
	--checkpoint_dir checkpoints/eliceiri_train \
	--batch_size 4 \
	> ./checkpoints/eliceiri_train/out.file 2>&1 &


# to be edited... 👇

# wait
### prepare test data
mkdir -p data/eliceiri/test
cp -r ../Datasets/Eliceiri_temp/A/test ./data/eliceiri/test/A
cp -r ../Datasets/Eliceiri_temp/B/test ./data/eliceiri/test/B
# ### Test
# # here input&output sizes are fixed to 1
# python test.py --dataroot ./data/eliceiri_star2_train/ --name eliceiri_train --model cycle_gan --num_test 99999 --batch_size 1 --preprocess pad --divisor 256 --input_nc 1 --output_nc 1 --gpu_ids $2

# ### unpad results
# python ../utils/unpad_results.py -p ./results/eliceiri_train/test_latest/images --width 600 --height 600

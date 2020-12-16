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


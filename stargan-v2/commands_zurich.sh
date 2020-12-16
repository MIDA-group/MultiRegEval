# Run: ./commands_zurich.sh {fold} {gpu_id}


## Zurich's data
### prepare training data
mkdir -p data/zurich/fold$1/train

### MUST RUN ON MIDA1!!! ###
ln -s /data2/jiahao/Registration/Datasets/Zurich_temp/fold$1/A/train data/zurich/fold$1/train/A
ln -s /data2/jiahao/Registration/Datasets/Zurich_temp/fold$1/B/train data/zurich/fold$1/train/B
ln -s /data2/jiahao/Registration/stargan-v2/data/zurich/fold$1/train data/zurich/fold$1/val
### MUST RUN ON MIDA1!!! ###

# cp -r ../Datasets/Zurich_temp/fold$1/A/train ./data/zurich/fold$1/train/A
# cp -r ../Datasets/Zurich_temp/fold$1/B/train ./data/zurich/fold$1/train/B
# use train as validation set
# cp -r data/zurich/fold$1/train data/zurich/fold$1/val


### Train
wait
# here input&output sizes are fixed to 1
mkdir -p ./checkpoints/zurich_train_fold$1
CUDA_VISIBLE_DEVICES=$2 nohup python main.py --mode train --img_size 256 --num_domains 2 --w_hpf 0 \
	--train_img_dir data/zurich/fold$1/train \
	--val_img_dir data/zurich/fold$1/val \
	--checkpoint_dir checkpoints/zurich_train_fold$1 \
	--batch_size 4 \
	> ./checkpoints/zurich_train_fold$1/out.file 2>&1 &
# 以上命令已运行


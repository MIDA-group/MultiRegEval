# Run: ./commands_eliceiri.sh {fold} {gpu_id}


## Eliceiri's data
### prepare training data
mkdir -p datasets/eliceiri/train
cp -r ../Datasets/Eliceiri_temp/A/train ./datasets/eliceiri/train/A
cp -r ../Datasets/Eliceiri_temp/B/train ./datasets/eliceiri/train/B
# use e0_* as validation set
mkdir -p datasets/eliceiri/val/A
mkdir -p datasets/eliceiri/val/B
mv ./datasets/eliceiri/train/A/e0_* datasets/eliceiri/val/A/
mv ./datasets/eliceiri/train/B/e0_* datasets/eliceiri/val/B/

### Train
wait
# here input&output sizes are fixed to 1
CUDA_VISIBLE_DEVICES=0 nohup python main.py --mode train --img_size 256 --num_domains 2 --w_hpf 0 --train_img_dir datasets/eliceiri/train --val_img_dir datasets/eliceiri/val --checkpoint_dir checkpoints/eliceiri_star2_train --batch_size 4 > ./checkpoints/eliceiri_star2_train/out.file 2>&1 &
# 以上命令已运行

# wait
### prepare test data
mkdir -p datasets/eliceiri/test
cp ./datasets/eliceiri/val/A/e0_b0_* ./datasets/eliceiri/test/A/
cp ./datasets/eliceiri/val/B/e0_b0_* ./datasets/eliceiri/test/B/
### Test
CUDA_VISIBLE_DEVICES=0 python main.py --mode test --num_domains 2 --w_hpf 0 --resume_iter 100000 \
               --checkpoint_dir checkpoints/eliceiri_star2_train \
               --result_dir expr/results/eliceiri_star2_trial2 \
               --src_dir datasets/eliceiri/some_patches \
               --ref_dir datasets/eliceiri/some_patches
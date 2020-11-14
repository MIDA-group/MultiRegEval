CUDA_VISIBLE_DEVICES=0 nohup python train_comir.py /data/johan/biodata/TrainSet2/SHG /data/johan/biodata/TrainSet2/BF -val_path_a /data/johan/biodata/Validation1Set/SHG -val_path_b /data/johan/biodata/Validation1Set/BF -log_a 1 -iterations 100 -l2 0.1 > ./logs/eliceiri_trial.file 2>&1 &


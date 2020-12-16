# Run: nohup ./commands_train.sh &

# Eliceiri's data

## Train
CUDA_VISIBLE_DEVICES=2 nohup python train_comir.py \
	../Datasets/Eliceiri_temp/A/train \
	../Datasets/Eliceiri_temp/B/train \
	-export_folder results/eliceiri_train \
	-log_a 1 -iterations 100 -l2 0.1 \
	> ./logs/eliceiri_train.file 2>&1 &
wait


# Zurich's data

## Train
for f in {1..3}; do
	CUDA_VISIBLE_DEVICES=2 nohup python train_comir.py \
		../Datasets/Zurich_temp/fold${f}/A/train \
		../Datasets/Zurich_temp/fold${f}/B/train \
		-export_folder results/zurich_train_fold${f} \
		-log_a 1 -iterations 100 -l2 0.1 \
		> ./logs/zurich_train_fold${f}.file 2>&1 &
	wait
done


# Balvan's data

## Train
for f in {1..3}; do
	CUDA_VISIBLE_DEVICES=2 nohup python train_comir.py \
		../Datasets/Balvan_temp/fold${f}/A/train \
		../Datasets/Balvan_temp/fold${f}/B/train \
		-export_folder results/balvan_train_fold${f} \
		-log_a 1 -iterations 100 -l2 0.1 \
		> ./logs/balvan_train_fold${f}.file 2>&1 &
	wait
done


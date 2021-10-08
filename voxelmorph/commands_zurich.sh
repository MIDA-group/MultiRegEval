# VoxelMorph
# nohup ./commands_zurich.sh &


mkdir ./logs


# For Zurich data
# dataname="zurich"
# fold="1"

# move and mix training data at all transformation level
for modality in A B; do
	tardir="./datasets/zurich/fold1/${modality}/train"
	mkdir -p ${tardir}
	for tlevel in {1..4}; do
		srcdir="../Datasets/Zurich_patches/fold1/patch_tlevel${tlevel}/${modality}/train"
		for i in `(cd ${srcdir} && ls *.png)`; do cp ${srcdir}/$i `echo "${tardir}/tlevel${tlevel}_"$i`; done
	done
done


## supervised
nohup python train_affine.py --data_root ./datasets/zurich/fold1 --a2b 0 --supervised --model_name zurich_fold1_su_b2a --epochs 1500 --batch_size 64 --gpu 0 > ./logs/zurich_fold1_su_b2a.out 2>&1 &

wait

## unsupervised
nohup python train_affine.py --data_root ./datasets/zurich/fold1 --a2b 0 --model_name zurich_fold1_us_b2a --epochs 1500 --batch_size 64 --gpu 1 > ./logs/zurich_fold1_us_b2a.out 2>&1 &

wait

## a2a trail
nohup python train_affine.py --data_root ./datasets/zurich/fold1 --supervised --model_name zurich_fold1_su_a2a --epochs 500 --steps_per_epoch 50 --batch_size 64 --gpu 2 > ./logs/zurich_fold1_su_a2a.out 2>&1 &


# small transformation trail
## supervised
nohup python train_affine.py --data_root ./datasets/zurich_tlevel1/fold1 --a2b 0 --supervised --model_name zurich_tlevel1_fold1_su_b2a --epochs 5000 --batch_size 64 --gpu 2 > ./logs/zurich_tlevel1_fold1_su_b2a.out 2>&1 &

## unsupervised
nohup python train_affine.py --data_root ./datasets/zurich_tlevel1/fold1 --a2b 0 --model_name zurich_tlevel1_fold1_us_b2a --epochs 5000 --batch_size 64 --gpu 1 > ./logs/zurich_tlevel1_fold1_us_b2a.out 2>&1 &

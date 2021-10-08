# VoxelMorph
# nohup ./commands_balvan.sh &


mkdir ./logs


# For Balvan's data
# dataname="balvan"
# fold="1"

# move and mix training data at all transformation level
for modality in A B; do
	tardir="./datasets/balvan/fold1/${modality}/train"
	mkdir -p ${tardir}
	for tlevel in {1..4}; do
		srcdir="../Datasets/Balvan_patches/fold1/patch_tlevel${tlevel}/${modality}/train"
		for i in `(cd ${srcdir} && ls *.png)`; do cp ${srcdir}/$i `echo "${tardir}/tlevel${tlevel}_"$i`; done
	done
done


## supervised
nohup python train_affine.py --data_root ./datasets/balvan/fold1 --a2b 0 --supervised --model_name balvan_fold1_su_b2a --epochs 1500 --batch_size 64 --gpu 0 > ./logs/balvan_fold1_su_b2a.out 2>&1 &

wait

## unsupervised
nohup python train_affine.py --data_root ./datasets/balvan/fold1 --a2b 0 --model_name balvan_fold1_us_b2a --epochs 1500 --batch_size 64 --gpu 1 > ./logs/balvan_fold1_us_b2a.out 2>&1 &

wait

## a2a trail
nohup python train_affine.py --data_root ./datasets/balvan/fold1 --supervised --model_name balvan_fold1_su_a2a --epochs 500 --steps_per_epoch 50 --batch_size 64 --gpu 2 > ./logs/balvan_fold1_su_a2a.out 2>&1 &


# small transformation trail
## supervised
nohup python train_affine.py --data_root ./datasets/balvan_tlevel1/fold1 --a2b 0 --supervised --model_name balvan_tlevel1_fold1_su_b2a --epochs 5000 --batch_size 64 --gpu 2 > ./logs/balvan_tlevel1_fold1_su_b2a.out 2>&1 &

## unsupervised
nohup python train_affine.py --data_root ./datasets/balvan_tlevel1/fold1 --a2b 0 --model_name balvan_tlevel1_fold1_us_b2a --epochs 5000 --batch_size 64 --gpu 1 > ./logs/balvan_tlevel1_fold1_us_b2a.out 2>&1 &

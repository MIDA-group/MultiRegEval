# VoxelMorph
# nohup ./commands_eliceiri.sh &


mkdir ./logs


# For Eliceiri's data
# dataname="eliceiri"
# fold="1"

# move and mix training data at all transformation level
for modality in A B; do
	tardir="./datasets/eliceiri/${modality}/train"
	mkdir -p ${tardir}
	for tlevel in {1..4}; do
		srcdir="../Datasets/Eliceiri_patches/patch_tlevel${tlevel}/${modality}/train"
		for i in `(cd ${srcdir} && ls *.tif)`; do cp ${srcdir}/$i `echo "${tardir}/tlevel${tlevel}_"$i`; done
	done
done


## supervised
nohup python train_affine.py --data_root ./datasets/eliceiri --a2b 0 --supervised --model_name eliceiri_su_b2a --epochs 1500 --steps_per_epoch 50 --batch_size 4 --gpu 2 > ./logs/eliceiri_su_b2a.out 2>&1 &

wait

## unsupervised
nohup python train_affine.py --data_root ./datasets/eliceiri --a2b 0 --model_name eliceiri_us_b2a --epochs 1500 --steps_per_epoch 50 --batch_size 4 --gpu 1 > ./logs/eliceiri_us_b2a.out 2>&1 &

# wait

## a2a trail
# nohup python train_affine.py --data_root ./datasets/eliceiri --supervised --model_name eliceiri_su_a2a --epochs 500 --steps_per_epoch 50 --batch_size 64 --gpu 2 > ./logs/eliceiri_su_a2a.out 2>&1 &
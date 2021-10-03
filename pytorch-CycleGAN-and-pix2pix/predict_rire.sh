# For RIRE's data (including rename_results.sh)
# Run: nohup ./predict_rire.sh {gpu_id} > ./checkpoints/predict_rire.out 2>&1 &

cd ../pytorch-CycleGAN-and-pix2pix

for f in {1..3}; do
	# pix2pix
	python test.py --dataroot ./datasets/rire_p2p_train/fold${f} --name rire_p2p_train_fold${f}_a2b --model pix2pix --num_test 99999 --direction AtoB --input_nc 1 --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids $1
	python test.py --dataroot ./datasets/rire_p2p_train/fold${f} --name rire_p2p_train_fold${f}_b2a --model pix2pix --num_test 99999 --direction BtoA --input_nc 1 --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids $1

	### rename_results
	mkdir -p ../Datasets/RIRE_slices_fake/fold${f}/p2p_A/
	mkdir -p ../Datasets/RIRE_slices_fake/fold${f}/p2p_B/
	cp ./results/rire_p2p_train_fold${f}_b2a/test_latest/images/*_fake_*.png ../Datasets/RIRE_slices_fake/fold${f}/p2p_A/
	cp ./results/rire_p2p_train_fold${f}_a2b/test_latest/images/*_fake_*.png ../Datasets/RIRE_slices_fake/fold${f}/p2p_B/
	rename -v '_fake_B' '' ../Datasets/RIRE_slices_fake/fold${f}/p2p_A/*.png
	rename -v '_fake_B' '' ../Datasets/RIRE_slices_fake/fold${f}/p2p_B/*.png
	

	# cycleGAN
	python test.py --dataroot ./datasets/rire_cyc_train/fold${f}/ --name rire_cyc_train_fold${f} --model cycle_gan --num_test 99999 --batch_size 4 --preprocess pad --divisor 256 --input_nc 1 --output_nc 1 --gpu_ids $1

	### rename_results
	mkdir -p ../Datasets/RIRE_slices_fake/fold${f}/cyc_A/
	mkdir -p ../Datasets/RIRE_slices_fake/fold${f}/cyc_B/
	cp ./results/rire_cyc_train_fold${f}/test_latest/images/*_fake_A.png ../Datasets/RIRE_slices_fake/fold${f}/cyc_A/
	cp ./results/rire_cyc_train_fold${f}/test_latest/images/*_fake_B.png ../Datasets/RIRE_slices_fake/fold${f}/cyc_B/
	rename -v '_fake_A' '' ../Datasets/RIRE_slices_fake/fold${f}/cyc_A/*.png
	rename -v '_fake_B' '' ../Datasets/RIRE_slices_fake/fold${f}/cyc_B/*.png

done

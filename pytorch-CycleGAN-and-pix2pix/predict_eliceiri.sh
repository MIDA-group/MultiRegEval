# For Eliceiri's data (including rename_results.sh)
# Run: nohup ./predict_eliceiri.sh {gpu_id} > ./checkpoints/predict_eliceiri.out 2>&1 &

# pix2pix
cd ../pytorch-CycleGAN-and-pix2pix

## Test 4-level data
### prepare
for tlevel in {1..4}; do
	python datasets/combine_A_and_B.py --fold_A ../Datasets/Eliceiri_patches/patch_tlevel${tlevel}/A --fold_B ../Datasets/Eliceiri_patches/patch_tlevel${tlevel}/B --fold_AB ./datasets/eliceiri_patches_p2p/tlevel${tlevel}
done

### test
for tlevel in {1..4}; do
	mkdir -p checkpoints/eliceiri_p2p_tlevel${tlevel}_a2b
	cp checkpoints/eliceiri_p2p_train_a2b/latest_net_* checkpoints/eliceiri_p2p_tlevel${tlevel}_a2b
	python test.py --dataroot ./datasets/eliceiri_patches_p2p/tlevel${tlevel} --name eliceiri_p2p_tlevel${tlevel}_a2b --model pix2pix --num_test 99999 --direction AtoB --input_nc 1 --output_nc 3 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids $1
	# # unpad results
	# python ../utils/unpad_results.py -p ./results/eliceiri_p2p_tlevel${tlevel}_a2b/test_latest/images --width 834 --height 834

	mkdir -p checkpoints/eliceiri_p2p_tlevel${tlevel}_b2a
	cp checkpoints/eliceiri_p2p_train_b2a/latest_net_* checkpoints/eliceiri_p2p_tlevel${tlevel}_b2a
	python test.py --dataroot ./datasets/eliceiri_patches_p2p/tlevel${tlevel} --name eliceiri_p2p_tlevel${tlevel}_b2a --model pix2pix --num_test 99999 --direction BtoA --input_nc 3 --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids $1
	# # unpad results
	# python ../utils/unpad_results.py -p ./results/eliceiri_p2p_tlevel${tlevel}_b2a/test_latest/images --width 834 --height 834
done

### rename_results
for tlevel in {1..4}; do
	mkdir -p ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/p2p_A/
	mkdir -p ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/p2p_B/
	cp ./results/eliceiri_p2p_tlevel${tlevel}_b2a/test_latest/images/*_fake_*.png ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/p2p_A/
	cp ./results/eliceiri_p2p_tlevel${tlevel}_a2b/test_latest/images/*_fake_*.png ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/p2p_B/
	rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/p2p_A/*.png
	rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/p2p_B/*.png
	# unpad results
	python ../utils/unpad_results.py -p ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/p2p_A --width 834 --height 834
	python ../utils/unpad_results.py -p ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/p2p_B --width 834 --height 834
done



# cycleGAN

## Test 4-level data
### prepare
for tlevel in {1..4}; do
	mkdir -p datasets/eliceiri_patches_cyc/tlevel${tlevel}
	cp -r ../Datasets/Eliceiri_patches/patch_tlevel${tlevel}/A/test/ ./datasets/eliceiri_patches_cyc/tlevel${tlevel}/testA
	cp -r ../Datasets/Eliceiri_patches/patch_tlevel${tlevel}/B/test/ ./datasets/eliceiri_patches_cyc/tlevel${tlevel}/testB
done

### test
for tlevel in {1..4}; do
	mkdir checkpoints/eliceiri_cyc_tlevel${tlevel}
	cp checkpoints/eliceiri_cyc_train/latest_net_* checkpoints/eliceiri_cyc_tlevel${tlevel}
	python test.py --dataroot ./datasets/eliceiri_patches_cyc/tlevel${tlevel}/ --name eliceiri_cyc_tlevel${tlevel} --model cycle_gan --num_test 99999 --batch_size 4 --preprocess pad --divisor 256 --gpu_ids $1

	# # unpad results
	# python ../utils/unpad_results.py -p ./results/eliceiri_cyc_tlevel${tlevel}/test_latest/images --width 834 --height 834
done

### rename_results
for tlevel in {1..4}; do
	mkdir -p ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/cyc_A/
	mkdir -p ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/cyc_B/
	cp ./results/eliceiri_cyc_tlevel${tlevel}/test_latest/images/*_fake_A.png ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/cyc_A/
	cp ./results/eliceiri_cyc_tlevel${tlevel}/test_latest/images/*_fake_B.png ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/cyc_B/
	rename -v '_fake_A' '' ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/cyc_A/*.png
	rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/cyc_B/*.png
	# unpad results
	python ../utils/unpad_results.py -p ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/cyc_A --width 834 --height 834
	python ../utils/unpad_results.py -p ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/cyc_B --width 834 --height 834
done
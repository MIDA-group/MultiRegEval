# For Zurich's data (including rename_results.sh)

# pix2pix
cd ../pytorch-CycleGAN-and-pix2pix

## Test 4-level data
### prepare
for tlevel in {1..4}; do
	python datasets/combine_A_and_B.py --fold_A ../Datasets/Zurich_patches/fold1/patch_tlevel${tlevel}/A --fold_B ../Datasets/Zurich_patches/fold1/patch_tlevel${tlevel}/B --fold_AB ./datasets/zurich_patches_p2p/fold1/tlevel${tlevel}
done

### test
for tlevel in {1..4}; do
	mkdir -p checkpoints/zurich_p2p_tlevel${tlevel}_fold1_a2b
	cp checkpoints/zurich_p2p_train_fold1_a2b/latest_net_* checkpoints/zurich_p2p_tlevel${tlevel}_fold1_a2b
	python test.py --dataroot ./datasets/zurich_patches_p2p/fold1/tlevel${tlevel} --name zurich_p2p_tlevel${tlevel}_fold1_a2b --model pix2pix --num_test 99999 --direction AtoB --input_nc 1 --output_nc 3 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 1
	# # unpad results
	# python ../utils/unpad_results.py -p ./results/zurich_p2p_tlevel${tlevel}_fold1_a2b/test_latest/images --width 300 --height 300

	mkdir -p checkpoints/zurich_p2p_tlevel${tlevel}_fold1_b2a
	cp checkpoints/zurich_p2p_train_fold1_b2a/latest_net_* checkpoints/zurich_p2p_tlevel${tlevel}_fold1_b2a
	python test.py --dataroot ./datasets/zurich_patches_p2p/fold1/tlevel${tlevel} --name zurich_p2p_tlevel${tlevel}_fold1_b2a --model pix2pix --num_test 99999 --direction BtoA --input_nc 3 --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 1
	# # unpad results
	# python ../utils/unpad_results.py -p ./results/zurich_p2p_tlevel${tlevel}_fold1_b2a/test_latest/images --width 300 --height 300
done

### rename_results
for tlevel in {1..4}; do
	mkdir -p ../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/p2p_A/
	mkdir -p ../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/p2p_B/
	cp ./results/zurich_p2p_tlevel${tlevel}_fold1_b2a/test_latest/images/*_fake_*.png ../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/p2p_A/
	cp ./results/zurich_p2p_tlevel${tlevel}_fold1_a2b/test_latest/images/*_fake_*.png ../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/p2p_B/
	rename -v '_fake_B' '' ../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/p2p_A/*.png
	rename -v '_fake_B' '' ../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/p2p_B/*.png
	# unpad results
	python ../utils/unpad_results.py -p ../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/p2p_A --width 300 --height 300
	python ../utils/unpad_results.py -p ../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/p2p_B --width 300 --height 300
done



# cycleGAN

## Test 4-level data
### prepare
for tlevel in {1..4}; do
	mkdir -p datasets/zurich_patches_cyc/fold1/tlevel${tlevel}
	cp -r ../Datasets/Zurich_patches/fold1/patch_tlevel${tlevel}/A/test/ ./datasets/zurich_patches_cyc/fold1/tlevel${tlevel}/testA
	cp -r ../Datasets/Zurich_patches/fold1/patch_tlevel${tlevel}/B/test/ ./datasets/zurich_patches_cyc/fold1/tlevel${tlevel}/testB
done

### test
for tlevel in {1..4}; do
	mkdir checkpoints/zurich_cyc_tlevel${tlevel}_fold1
	cp checkpoints/zurich_cyc_train_fold1/latest_net_* checkpoints/zurich_cyc_tlevel${tlevel}_fold1
	python test.py --dataroot ./datasets/zurich_patches_cyc/fold1/tlevel${tlevel}/ --name zurich_cyc_tlevel${tlevel}_fold1 --model cycle_gan --num_test 99999 --batch_size 4 --preprocess pad --divisor 256 --gpu_ids 1

	# # unpad results
	# python ../utils/unpad_results.py -p ./results/zurich_cyc_tlevel${tlevel}_fold1/test_latest/images --width 300 --height 300
done

### rename_results
for tlevel in {1..4}; do
	mkdir -p ../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/cyc_A/
	mkdir -p ../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/cyc_B/
	cp ./results/zurich_cyc_tlevel${tlevel}_fold1/test_latest/images/*_fake_A.png ../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/cyc_A/
	cp ./results/zurich_cyc_tlevel${tlevel}_fold1/test_latest/images/*_fake_B.png ../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/cyc_B/
	rename -v '_fake_A' '' ../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/cyc_A/*.png
	rename -v '_fake_B' '' ../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/cyc_B/*.png
	# unpad results
	python ../utils/unpad_results.py -p ../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/cyc_A --width 300 --height 300
	python ../utils/unpad_results.py -p ../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/cyc_B --width 300 --height 300
done
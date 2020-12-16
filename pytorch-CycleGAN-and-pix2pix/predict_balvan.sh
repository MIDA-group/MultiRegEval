# For Balvan's data (including rename_results.sh)
# Run: nohup ./predict_balvan.sh {gpu_id} > ./checkpoints/predict_balvan.out 2>&1 &

# pix2pix
cd ../pytorch-CycleGAN-and-pix2pix

for f in {1..3}; do
	## Test 4-level data
	### prepare
	for tlevel in {1..4}; do
		python datasets/combine_A_and_B.py --fold_A ../Datasets/Balvan_patches/fold${f}/patch_tlevel${tlevel}/A --fold_B ../Datasets/Balvan_patches/fold${f}/patch_tlevel${tlevel}/B --fold_AB ./datasets/balvan_patches_p2p/fold${f}/tlevel${tlevel}
	done

	### test
	for tlevel in {1..4}; do
		mkdir -p checkpoints/balvan_p2p_tlevel${tlevel}_fold${f}_a2b
		cp checkpoints/balvan_p2p_train_fold${f}_a2b/latest_net_* checkpoints/balvan_p2p_tlevel${tlevel}_fold${f}_a2b
		python test.py --dataroot ./datasets/balvan_patches_p2p/fold${f}/tlevel${tlevel} --name balvan_p2p_tlevel${tlevel}_fold${f}_a2b --model pix2pix --num_test 99999 --direction AtoB --input_nc 1 --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids $1
		# # unpad results
		# python ../utils/unpad_results.py -p ./results/balvan_p2p_tlevel${tlevel}_fold${f}_a2b/test_latest/images --width 300 --height 300

		mkdir -p checkpoints/balvan_p2p_tlevel${tlevel}_fold${f}_b2a
		cp checkpoints/balvan_p2p_train_fold${f}_b2a/latest_net_* checkpoints/balvan_p2p_tlevel${tlevel}_fold${f}_b2a
		python test.py --dataroot ./datasets/balvan_patches_p2p/fold${f}/tlevel${tlevel} --name balvan_p2p_tlevel${tlevel}_fold${f}_b2a --model pix2pix --num_test 99999 --direction BtoA --input_nc 1 --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids $1
		# # unpad results
		# python ../utils/unpad_results.py -p ./results/balvan_p2p_tlevel${tlevel}_fold${f}_b2a/test_latest/images --width 300 --height 300
	done

	### rename_results
	for tlevel in {1..4}; do
		mkdir -p ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/p2p_A/
		mkdir -p ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/p2p_B/
		cp ./results/balvan_p2p_tlevel${tlevel}_fold${f}_b2a/test_latest/images/*_fake_*.png ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/p2p_A/
		cp ./results/balvan_p2p_tlevel${tlevel}_fold${f}_a2b/test_latest/images/*_fake_*.png ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/p2p_B/
		rename -v '_fake_B' '' ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/p2p_A/*.png
		rename -v '_fake_B' '' ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/p2p_B/*.png
		# unpad results
		python ../utils/unpad_results.py -p ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/p2p_A --width 300 --height 300
		python ../utils/unpad_results.py -p ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/p2p_B --width 300 --height 300
	done



	# cycleGAN

	## Test 4-level data
	### prepare
	for tlevel in {1..4}; do
		mkdir -p datasets/balvan_patches_cyc/fold${f}/tlevel${tlevel}
		cp -r ../Datasets/Balvan_patches/fold${f}/patch_tlevel${tlevel}/A/test/ ./datasets/balvan_patches_cyc/fold${f}/tlevel${tlevel}/testA
		cp -r ../Datasets/Balvan_patches/fold${f}/patch_tlevel${tlevel}/B/test/ ./datasets/balvan_patches_cyc/fold${f}/tlevel${tlevel}/testB
	done

	### test
	for tlevel in {1..4}; do
		mkdir checkpoints/balvan_cyc_tlevel${tlevel}_fold${f}
		cp checkpoints/balvan_cyc_train_fold${f}/latest_net_* checkpoints/balvan_cyc_tlevel${tlevel}_fold${f}
		python test.py --dataroot ./datasets/balvan_patches_cyc/fold${f}/tlevel${tlevel}/ --name balvan_cyc_tlevel${tlevel}_fold${f} --model cycle_gan --num_test 99999 --batch_size 4 --preprocess pad --divisor 256 --input_nc 1 --output_nc 1 --gpu_ids $1

		# # unpad results
		# python ../utils/unpad_results.py -p ./results/balvan_cyc_tlevel${tlevel}_fold${f}/test_latest/images --width 300 --height 300
	done

	### rename_results
	for tlevel in {1..4}; do
		mkdir -p ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/cyc_A/
		mkdir -p ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/cyc_B/
		cp ./results/balvan_cyc_tlevel${tlevel}_fold${f}/test_latest/images/*_fake_A.png ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/cyc_A/
		cp ./results/balvan_cyc_tlevel${tlevel}_fold${f}/test_latest/images/*_fake_B.png ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/cyc_B/
		rename -v '_fake_A' '' ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/cyc_A/*.png
		rename -v '_fake_B' '' ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/cyc_B/*.png
		# unpad results
		python ../utils/unpad_results.py -p ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/cyc_A --width 300 --height 300
		python ../utils/unpad_results.py -p ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/cyc_B --width 300 --height 300
	done
done

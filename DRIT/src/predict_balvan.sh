# DRIT++ for Balvan's data (including rename_results.sh)
# Run: nohup ./predict_balvan.sh > ./outfiles/predict_balvan.out 2>&1 &

for f in {1..3}; do
	## Test 4-level
	for tlevel in {1..4}; do
		python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/balvan_patches_cyc/fold${f}/tlevel${tlevel}/ --resize_size 300 --crop_size 300 --a2b 1 --name balvan_drit_tlevel${tlevel}_fold${f}_a2b --concat 0 --resume ../results/balvan_drit_train_fold${f}/01199.pth --input_dim_a 1 --input_dim_b 1 
		# strangely, --gpu can only be 0
		python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/balvan_patches_cyc/fold${f}/tlevel${tlevel}/ --resize_size 300 --crop_size 300 --a2b 0 --name balvan_drit_tlevel${tlevel}_fold${f}_b2a --concat 0 --resume ../results/balvan_drit_train_fold${f}/01199.pth --input_dim_a 1 --input_dim_b 1 
	done

	### rename_results
	for tlevel in {1..4}; do
		mkdir -p ../../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/drit_A/
		mkdir -p ../../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/drit_B/
		cp ../outputs/balvan_drit_tlevel${tlevel}_fold${f}_b2a/*_fake.png ../../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/drit_A/
		cp ../outputs/balvan_drit_tlevel${tlevel}_fold${f}_a2b/*_fake.png ../../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/drit_B/
		rename -v '_fake' '' ../../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/drit_A/*.png
		rename -v '_fake' '' ../../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/drit_B/*.png
	done
done

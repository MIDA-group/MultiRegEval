# DRIT++ for Zurich's data (including rename_results.sh)
cd ../DRIT/src/

## Test 4-level
for tlevel in {1..4}; do
	python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/zurich_patches_cyc/fold1/tlevel${tlevel}/ --resize_size 300 --crop_size 300 --a2b 1 --name zurich_drit_tlevel${tlevel}_fold1_a2b --concat 1 --resume ../results/zurich_drit_train_fold1/01199.pth
	# strangely, --gpu can only be 0
	python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/zurich_patches_cyc/fold1/tlevel${tlevel}/ --resize_size 300 --crop_size 300 --a2b 0 --name zurich_drit_tlevel${tlevel}_fold1_b2a --concat 1 --resume ../results/zurich_drit_train_fold1/01199.pth
done

### rename_results
for tlevel in {1..4}; do
	mkdir -p ../../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/drit_A/
	mkdir -p ../../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/drit_B/
	cp ../outputs/zurich_drit_tlevel${tlevel}_fold1_b2a/*_fake.png ../../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/drit_A/
	cp ../outputs/zurich_drit_tlevel${tlevel}_fold1_a2b/*_fake.png ../../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/drit_B/
	rename -v '_fake' '' ../../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/drit_A/*.png
	rename -v '_fake' '' ../../Datasets/Zurich_patches_fake/fold1/patch_tlevel${tlevel}/drit_B/*.png
done

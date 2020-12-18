# DRIT++ for Eliceiri's data (including rename_results.sh)
# Run: nohup ./predict_eliceiri.sh > ./outfiles/predict_eliceiri.out 2>&1 &

## Test 4-level
for tlevel in {1..4}; do
	python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_patches_cyc/tlevel${tlevel}/ --resize_size 834 --crop_size 834 --a2b 1 --name eliceiri_drit_tlevel${tlevel}_a2b --concat 0 --resume ../results/eliceiri_drit_train/01199.pth
	# strangely, --gpu can only be 0
	python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_patches_cyc/tlevel${tlevel}/ --resize_size 834 --crop_size 834 --a2b 0 --name eliceiri_drit_tlevel${tlevel}_b2a --concat 0 --resume ../results/eliceiri_drit_train/01199.pth
done

### rename_results
for tlevel in {1..4}; do
	mkdir -p ../../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/drit_A/
	mkdir -p ../../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/drit_B/
	cp ../outputs/eliceiri_drit_tlevel${tlevel}_b2a/*_fake.png ../../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/drit_A/
	cp ../outputs/eliceiri_drit_tlevel${tlevel}_a2b/*_fake.png ../../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/drit_B/
	rename -v '_fake' '' ../../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/drit_A/*.png
	rename -v '_fake' '' ../../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/drit_B/*.png
done

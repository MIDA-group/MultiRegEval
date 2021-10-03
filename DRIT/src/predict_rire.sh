# DRIT++ for RIRE data (including rename_results.sh)
# Run: nohup ./predict_rire.sh > ./outfiles/predict_rire.out 2>&1 &

# clean previous outputs
# rm -r ../datasets/rire_train/
# rm -r ../outputs/rire_drit_train_fold*/
# rm -r ../../Datasets/RIRE_slices_fake/fold*/drit_*/
# rm -r ../../Datasets/RIRE_patches_fake/fold*/patch_tlevel*/drit_*/
# rm ./outfiles/predict_rire.out

for f in {1..3}; do
	python ../../utils/pad_images.py -i ../../Datasets/RIRE_temp/fold${f}/A/test -o ../datasets/rire_train/fold${f}/testA -d 512
	python ../../utils/pad_images.py -i ../../Datasets/RIRE_temp/fold${f}/B/test -o ../datasets/rire_train/fold${f}/testB -d 512

	python test_transfer.py --dataroot ../datasets/rire_train/fold${f}/ --resize_size 512 --crop_size 512 --a2b 1 --name rire_drit_train_fold${f}_a2b --concat 0 --resume ../results/rire_drit_train_fold${f}/01199.pth --input_dim_a 1 --input_dim_b 1 
	# strangely, --gpu can only be 0
	python test_transfer.py --dataroot ../datasets/rire_train/fold${f}/ --resize_size 512 --crop_size 512 --a2b 0 --name rire_drit_train_fold${f}_b2a --concat 0 --resume ../results/rire_drit_train_fold${f}/01199.pth --input_dim_a 1 --input_dim_b 1 

	### rename_results
	mkdir -p ../../Datasets/RIRE_slices_fake/fold${f}/drit_A/
	mkdir -p ../../Datasets/RIRE_slices_fake/fold${f}/drit_B/
	cp ../outputs/rire_drit_train_fold${f}_b2a/*_fake.png ../../Datasets/RIRE_slices_fake/fold${f}/drit_A/
	cp ../outputs/rire_drit_train_fold${f}_a2b/*_fake.png ../../Datasets/RIRE_slices_fake/fold${f}/drit_B/
	rename -v '_fake' '' ../../Datasets/RIRE_slices_fake/fold${f}/drit_A/*.png
	rename -v '_fake' '' ../../Datasets/RIRE_slices_fake/fold${f}/drit_B/*.png
done

# Eliceiri's data
# Run: nohup ./predict_eliceiri.sh > ./checkpoints/predict_eliceiri.out 2>&1 &


### prepare evaluation data
for tlevel in {1..4}; do
	mkdir -p data/eliceiri/test/tlevel${tlevel}
	### MUST RUN ON MIDA1!!! ###
	ln -s /data2/jiahao/Registration/Datasets/Eliceiri_patches/patch_tlevel${tlevel}/A/test data/eliceiri/test/tlevel${tlevel}/A
	ln -s /data2/jiahao/Registration/Datasets/Eliceiri_patches/patch_tlevel${tlevel}/B/test data/eliceiri/test/tlevel${tlevel}/B
	### MUST RUN ON MIDA1!!! ###
done


### predict and rename_results
for tlevel in {1..4}; do
	CUDA_VISIBLE_DEVICES=0 python main.py --mode test --num_domains 2 --w_hpf 0 --resume_iter 100000 \
	               --checkpoint_dir checkpoints/eliceiri_train \
	               --result_dir results/eliceiri_patches_star/tlevel${tlevel} \
	               --src_dir data/eliceiri/test/tlevel${tlevel} \
	               --ref_dir data/eliceiri/test/tlevel${tlevel}

	mkdir -p ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/star_A/
	mkdir -p ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/star_B/
	cp ./results/eliceiri_patches_star/tlevel${tlevel}/*_fake_A.png ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/star_A/
	cp ./results/eliceiri_patches_star/tlevel${tlevel}/*_fake_B.png ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/star_B/
	rename -v '_fake_A' '' ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/star_A/*.png
	rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/star_B/*.png
done
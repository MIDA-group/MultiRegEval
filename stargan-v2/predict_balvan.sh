# Balvan's data
# Run: nohup ./predict_balvan.sh > ./checkpoints/predict_balvan.out 2>&1 &


### prepare evaluation data
for f in {1..3}; do
	for tlevel in {1..4}; do
		mkdir -p data/balvan/fold${f}/test/tlevel${tlevel}
		### MUST RUN ON MIDA1!!! ###
		ln -s /data2/jiahao/Registration/Datasets/Balvan_patches/fold${f}/patch_tlevel${tlevel}/A/test data/balvan/fold${f}/test/tlevel${tlevel}/A
		ln -s /data2/jiahao/Registration/Datasets/Balvan_patches/fold${f}/patch_tlevel${tlevel}/B/test data/balvan/fold${f}/test/tlevel${tlevel}/B
		### MUST RUN ON MIDA1!!! ###
	done
done


### predict and rename_results
for f in {1..3}; do
	for tlevel in {1..4}; do
		CUDA_VISIBLE_DEVICES=0 python main.py --mode test --num_domains 2 --w_hpf 0 --resume_iter 100000 \
		               --checkpoint_dir checkpoints/balvan_train_fold${f} \
		               --result_dir results/balvan_patches_star_fold${f}/tlevel${tlevel} \
		               --src_dir data/balvan/fold${f}/test/tlevel${tlevel} \
		               --ref_dir data/balvan/fold${f}/test/tlevel${tlevel}

		mkdir -p ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/star_A/
		mkdir -p ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/star_B/
		cp ./results/balvan_patches_star_fold${f}/tlevel${tlevel}/*_fake_A.png ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/star_A/
		cp ./results/balvan_patches_star_fold${f}/tlevel${tlevel}/*_fake_B.png ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/star_B/
		rename -v '_fake_A' '' ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/star_A/*.png
		rename -v '_fake_B' '' ../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/star_B/*.png
	done
done

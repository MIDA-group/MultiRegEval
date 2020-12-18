# Zurich's data
# Run: nohup ./predict_zurich.sh > ./checkpoints/predict_zurich.out 2>&1 &


### prepare evaluation data
for f in {1..3}; do
	for tlevel in {1..4}; do
		mkdir -p data/zurich/fold${f}/test/tlevel${tlevel}
		### MUST RUN ON MIDA1!!! ###
		ln -s /data2/jiahao/Registration/Datasets/Zurich_patches/fold${f}/patch_tlevel${tlevel}/A/test data/zurich/fold${f}/test/tlevel${tlevel}/A
		ln -s /data2/jiahao/Registration/Datasets/Zurich_patches/fold${f}/patch_tlevel${tlevel}/B/test data/zurich/fold${f}/test/tlevel${tlevel}/B
		### MUST RUN ON MIDA1!!! ###
	done
done


### predict and rename_results
for f in {1..3}; do
	for tlevel in {1..4}; do
		CUDA_VISIBLE_DEVICES=0 python main.py --mode test --num_domains 2 --w_hpf 0 --resume_iter 100000 \
		               --checkpoint_dir checkpoints/zurich_train_fold${f} \
		               --result_dir results/zurich_patches_star_fold${f}/tlevel${tlevel} \
		               --src_dir data/zurich/fold${f}/test/tlevel${tlevel} \
		               --ref_dir data/zurich/fold${f}/test/tlevel${tlevel}

		mkdir -p ../Datasets/Zurich_patches_fake/fold${f}/patch_tlevel${tlevel}/star_A/
		mkdir -p ../Datasets/Zurich_patches_fake/fold${f}/patch_tlevel${tlevel}/star_B/
		cp ./results/zurich_patches_star_fold${f}/tlevel${tlevel}/*_fake_A.png ../Datasets/Zurich_patches_fake/fold${f}/patch_tlevel${tlevel}/star_A/
		cp ./results/zurich_patches_star_fold${f}/tlevel${tlevel}/*_fake_B.png ../Datasets/Zurich_patches_fake/fold${f}/patch_tlevel${tlevel}/star_B/
		rename -v '_fake_A' '' ../Datasets/Zurich_patches_fake/fold${f}/patch_tlevel${tlevel}/star_A/*.png
		rename -v '_fake_B' '' ../Datasets/Zurich_patches_fake/fold${f}/patch_tlevel${tlevel}/star_B/*.png
		# unpad results
		python ../utils/unpad_results.py -p ../Datasets/Zurich_patches_fake/fold${f}/patch_tlevel${tlevel}/star_A --width 300 --height 300
		python ../utils/unpad_results.py -p ../Datasets/Zurich_patches_fake/fold${f}/patch_tlevel${tlevel}/star_B --width 300 --height 300
	done
done

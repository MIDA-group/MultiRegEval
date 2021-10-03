# RIRE data
# Run: nohup ./predict_rire.sh > ./checkpoints/predict_rire.out 2>&1 &


### prepare evaluation data
for f in {1..3}; do
	mkdir -p data/rire/fold${f}/test
	### MUST RUN ON MIDA1!!! ###
	ln -s /data2/jiahao/Registration/Datasets/RIRE_temp/fold${f}/A/test data/rire/fold${f}/test/A
	ln -s /data2/jiahao/Registration/Datasets/RIRE_temp/fold${f}/B/test data/rire/fold${f}/test/B
	### MUST RUN ON MIDA1!!! ###
done


### predict and rename_results
for f in {1..3}; do
	CUDA_VISIBLE_DEVICES=0 python main.py --mode test --num_domains 2 --w_hpf 0 --resume_iter 100000 \
	               --checkpoint_dir checkpoints/rire_train_fold${f} \
	               --result_dir results/rire_star_fold${f} \
	               --src_dir data/rire/fold${f}/test \
	               --ref_dir data/rire/fold${f}/test

	mkdir -p ../Datasets/RIRE_slices_fake/fold${f}/star_A/
	mkdir -p ../Datasets/RIRE_slices_fake/fold${f}/star_B/
	cp ./results/rire_star_fold${f}/*_fake_A.png ../Datasets/RIRE_slices_fake/fold${f}/star_A/
	cp ./results/rire_star_fold${f}/*_fake_B.png ../Datasets/RIRE_slices_fake/fold${f}/star_B/
	rename -v '_fake_A' '' ../Datasets/RIRE_slices_fake/fold${f}/star_A/*.png
	rename -v '_fake_B' '' ../Datasets/RIRE_slices_fake/fold${f}/star_B/*.png
	# # unpad results
	# python ../utils/unpad_results.py -p ../Datasets/RIRE_slices_fake/fold${f}/star_A --width 300 --height 300
	# python ../utils/unpad_results.py -p ../Datasets/RIRE_slices_fake/fold${f}/star_B --width 300 --height 300
done

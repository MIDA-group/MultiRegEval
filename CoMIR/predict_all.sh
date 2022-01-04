# CoMIR representations for all data
# Run: nohup ./predict_all.sh {gpu_id} > ./logs/predict_all.out 2>&1 &

# # Eliceiri's data
# for tlevel in {1..4}; do
# 	CUDA_VISIBLE_DEVICES=$1 python inference_comir.py \
# 		./results/eliceiri_train/latest.pt \
# 		../Datasets/Eliceiri_patches/patch_tlevel${tlevel}/A/test/ \
# 		../Datasets/Eliceiri_patches/patch_tlevel${tlevel}/B/test/ \
# 		../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/comir_A \
# 		../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/comir_B
# done


# # Balvan's data
# for f in {1..3}; do
# 	for tlevel in {1..4}; do
# 		CUDA_VISIBLE_DEVICES=$1 python inference_comir.py \
# 			./results/balvan_train_fold${f}/latest.pt \
# 			../Datasets/Balvan_patches/fold${f}/patch_tlevel${tlevel}/A/test/ \
# 			../Datasets/Balvan_patches/fold${f}/patch_tlevel${tlevel}/B/test/ \
# 			../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/comir_A \
# 			../Datasets/Balvan_patches_fake/fold${f}/patch_tlevel${tlevel}/comir_B
# 	done
# done


# # Zurich's data
# for f in {1..3}; do
# 	for tlevel in {1..4}; do
# 		CUDA_VISIBLE_DEVICES=$1 python inference_comir.py \
# 			./results/zurich_train_fold${f}/latest.pt \
# 			../Datasets/Zurich_patches/fold${f}/patch_tlevel${tlevel}/A/test/ \
# 			../Datasets/Zurich_patches/fold${f}/patch_tlevel${tlevel}/B/test/ \
# 			../Datasets/Zurich_patches_fake/fold${f}/patch_tlevel${tlevel}/comir_A \
# 			../Datasets/Zurich_patches_fake/fold${f}/patch_tlevel${tlevel}/comir_B
# 	done
# done

# RIRE data
for f in {1..3}; do
	CUDA_VISIBLE_DEVICES=$1 python inference_comir.py \
		./results/rire_train_redo128t10_fold${f}/latest.pt \
		../Datasets/RIRE_temp/fold${f}/A/test/ \
		../Datasets/RIRE_temp/fold${f}/B/test/ \
		../Datasets/RIRE_slices_fake/fold${f}/comir_redo128t10_A \
		../Datasets/RIRE_slices_fake/fold${f}/comir_redo128t10_B
done




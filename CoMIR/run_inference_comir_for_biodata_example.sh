# CUDA_VISIBLE_DEVICES=0 python inference_comir.py ./results/latest.pt ../Datasets/Eliceiri_test/processed/A/test ../Datasets/Eliceiri_test/processed/B/test ./outputs/eliceiri_trial/A ./outputs/eliceiri_trial/B

for tlevel in {1..4}; do
	CUDA_VISIBLE_DEVICES=0 python inference_comir.py \
		./results/Model1.pt \
		../Datasets/Eliceiri_patches/patch_tlevel${tlevel}/A/test/ \
		../Datasets/Eliceiri_patches/patch_tlevel${tlevel}/B/test/ \
		../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/comir1_A \
		../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/comir1_B
done

for tlevel in {1..4}; do
	CUDA_VISIBLE_DEVICES=0 python inference_comir.py \
		./results/Model2.pt \
		../Datasets/Eliceiri_patches/patch_tlevel${tlevel}/A/test/ \
		../Datasets/Eliceiri_patches/patch_tlevel${tlevel}/B/test/ \
		../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/comir2_A \
		../Datasets/Eliceiri_patches_fake/patch_tlevel${tlevel}/comir2_B
done


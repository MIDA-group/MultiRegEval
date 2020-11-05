# For Eliceiri's data 
# clean results
## DRIT
mkdir -p ../../Datasets/Eliceiri_patches_fake/processed/drit_A/
mkdir -p ../../Datasets/Eliceiri_patches_fake/processed/drit_B/
cp ../outputs/eliceiri_drit_processed_b2a/*_fake.png ../../Datasets/Eliceiri_patches_fake/processed/drit_A/
cp ../outputs/eliceiri_drit_processed_a2b/*_fake.png ../../Datasets/Eliceiri_patches_fake/processed/drit_B/
rename -v '_fake' '' ../../Datasets/Eliceiri_patches_fake/processed/drit_A/*.png
rename -v '_fake' '' ../../Datasets/Eliceiri_patches_fake/processed/drit_B/*.png

mkdir -p ../../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/drit_A/
mkdir -p ../../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/drit_B/
cp ../outputs/eliceiri_drit_rot0-5_b2a/*_fake.png ../../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/drit_A/
cp ../outputs/eliceiri_drit_rot0-5_a2b/*_fake.png ../../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/drit_B/
rename -v '_fake' '' ../../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/drit_A/*.png
rename -v '_fake' '' ../../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/drit_B/*.png

mkdir -p ../../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/drit_A/
mkdir -p ../../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/drit_B/
cp ../outputs/eliceiri_drit_rot5-10_b2a/*_fake.png ../../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/drit_A/
cp ../outputs/eliceiri_drit_rot5-10_a2b/*_fake.png ../../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/drit_B/
rename -v '_fake' '' ../../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/drit_A/*.png
rename -v '_fake' '' ../../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/drit_B/*.png

mkdir -p ../../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/drit_A/
mkdir -p ../../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/drit_B/
cp ../outputs/eliceiri_drit_rot10-15_b2a/*_fake.png ../../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/drit_A/
cp ../outputs/eliceiri_drit_rot10-15_a2b/*_fake.png ../../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/drit_B/
rename -v '_fake' '' ../../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/drit_A/*.png
rename -v '_fake' '' ../../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/drit_B/*.png

mkdir -p ../../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/drit_A/
mkdir -p ../../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/drit_B/
cp ../outputs/eliceiri_drit_rot15-20_b2a/*_fake.png ../../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/drit_A/
cp ../outputs/eliceiri_drit_rot15-20_a2b/*_fake.png ../../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/drit_B/
rename -v '_fake' '' ../../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/drit_A/*.png
rename -v '_fake' '' ../../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/drit_B/*.png

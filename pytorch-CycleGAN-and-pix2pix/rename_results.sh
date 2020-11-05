# For Eliceiri's data 
# clean results
## pix2pix
mkdir -p ../Datasets/Eliceiri_patches_fake/processed/p2p_A/
mkdir -p ../Datasets/Eliceiri_patches_fake/processed/p2p_B/
cp ./results/eliceiri_p2p_processed_b2a/test_latest/images/*_fake_*.png ../Datasets/Eliceiri_patches_fake/processed/p2p_A/
cp ./results/eliceiri_p2p_processed_a2b/test_latest/images/*_fake_*.png ../Datasets/Eliceiri_patches_fake/processed/p2p_B/
rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/processed/p2p_A/*.png
rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/processed/p2p_B/*.png

mkdir -p ../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/p2p_A/
mkdir -p ../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/p2p_B/
cp ./results/eliceiri_p2p_rot0-5_b2a/test_latest/images/*_fake_*.png ../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/p2p_A/
cp ./results/eliceiri_p2p_rot0-5_a2b/test_latest/images/*_fake_*.png ../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/p2p_B/
rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/p2p_A/*.png
rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/p2p_B/*.png

mkdir -p ../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/p2p_A/
mkdir -p ../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/p2p_B/
cp ./results/eliceiri_p2p_rot5-10_b2a/test_latest/images/*_fake_*.png ../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/p2p_A/
cp ./results/eliceiri_p2p_rot5-10_a2b/test_latest/images/*_fake_*.png ../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/p2p_B/
rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/p2p_A/*.png
rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/p2p_B/*.png

mkdir -p ../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/p2p_A/
mkdir -p ../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/p2p_B/
cp ./results/eliceiri_p2p_rot10-15_b2a/test_latest/images/*_fake_*.png ../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/p2p_A/
cp ./results/eliceiri_p2p_rot10-15_a2b/test_latest/images/*_fake_*.png ../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/p2p_B/
rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/p2p_A/*.png
rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/p2p_B/*.png

mkdir -p ../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/p2p_A/
mkdir -p ../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/p2p_B/
cp ./results/eliceiri_p2p_rot15-20_b2a/test_latest/images/*_fake_*.png ../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/p2p_A/
cp ./results/eliceiri_p2p_rot15-20_a2b/test_latest/images/*_fake_*.png ../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/p2p_B/
rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/p2p_A/*.png
rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/p2p_B/*.png

## cycleGAN
mkdir -p ../Datasets/Eliceiri_patches_fake/processed/cyc_A/
mkdir -p ../Datasets/Eliceiri_patches_fake/processed/cyc_B/
cp ./results/eliceiri_cyc_processed/test_latest/images/*_fake_A.png ../Datasets/Eliceiri_patches_fake/processed/cyc_A/
cp ./results/eliceiri_cyc_processed/test_latest/images/*_fake_B.png ../Datasets/Eliceiri_patches_fake/processed/cyc_B/
rename -v '_fake_A' '' ../Datasets/Eliceiri_patches_fake/processed/cyc_A/*.png
rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/processed/cyc_B/*.png

mkdir -p ../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/cyc_A/
mkdir -p ../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/cyc_B/
cp ./results/eliceiri_cyc_rot0-5/test_latest/images/*_fake_A.png ../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/cyc_A/
cp ./results/eliceiri_cyc_rot0-5/test_latest/images/*_fake_B.png ../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/cyc_B/
rename -v '_fake_A' '' ../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/cyc_A/*.png
rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/patch_trans0-20_rot0-5/cyc_B/*.png

mkdir -p ../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/cyc_A/
mkdir -p ../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/cyc_B/
cp ./results/eliceiri_cyc_rot5-10/test_latest/images/*_fake_A.png ../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/cyc_A/
cp ./results/eliceiri_cyc_rot5-10/test_latest/images/*_fake_B.png ../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/cyc_B/
rename -v '_fake_A' '' ../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/cyc_A/*.png
rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/patch_trans20-40_rot5-10/cyc_B/*.png

mkdir -p ../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/cyc_A/
mkdir -p ../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/cyc_B/
cp ./results/eliceiri_cyc_rot10-15/test_latest/images/*_fake_A.png ../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/cyc_A/
cp ./results/eliceiri_cyc_rot10-15/test_latest/images/*_fake_B.png ../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/cyc_B/
rename -v '_fake_A' '' ../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/cyc_A/*.png
rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/patch_trans40-60_rot10-15/cyc_B/*.png

mkdir -p ../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/cyc_A/
mkdir -p ../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/cyc_B/
cp ./results/eliceiri_cyc_rot15-20/test_latest/images/*_fake_A.png ../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/cyc_A/
cp ./results/eliceiri_cyc_rot15-20/test_latest/images/*_fake_B.png ../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/cyc_B/
rename -v '_fake_A' '' ../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/cyc_A/*.png
rename -v '_fake_B' '' ../Datasets/Eliceiri_patches_fake/patch_trans60-80_rot15-20/cyc_B/*.png


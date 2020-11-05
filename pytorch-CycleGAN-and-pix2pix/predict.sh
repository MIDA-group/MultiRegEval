# For Eliceiri's data 

# pix2pix

## Test 4-level data
### prepare
python datasets/combine_A_and_B.py --fold_A ../Datasets/Eliceiri_patches/patch_trans0-20_rot0-5/A --fold_B ../Datasets/Eliceiri_patches/patch_trans0-20_rot0-5/B --fold_AB ./datasets/eliceiri_patches_p2p/rot0-5
python datasets/combine_A_and_B.py --fold_A ../Datasets/Eliceiri_patches/patch_trans20-40_rot5-10/A --fold_B ../Datasets/Eliceiri_patches/patch_trans20-40_rot5-10/B --fold_AB ./datasets/eliceiri_patches_p2p/rot5-10
python datasets/combine_A_and_B.py --fold_A ../Datasets/Eliceiri_patches/patch_trans40-60_rot10-15/A --fold_B ../Datasets/Eliceiri_patches/patch_trans40-60_rot10-15/B --fold_AB ./datasets/eliceiri_patches_p2p/rot10-15
python datasets/combine_A_and_B.py --fold_A ../Datasets/Eliceiri_patches/patch_trans60-80_rot15-20/A --fold_B ../Datasets/Eliceiri_patches/patch_trans60-80_rot15-20/B --fold_AB ./datasets/eliceiri_patches_p2p/rot15-20

# level 1
mkdir checkpoints/eliceiri_p2p_rot0-5_a2b
cp checkpoints/eliceiri_p2p_train_a2b/latest_net_* checkpoints/eliceiri_p2p_rot0-5_a2b
python test.py --dataroot ./datasets/eliceiri_patches_p2p/rot0-5 --name eliceiri_p2p_rot0-5_a2b --model pix2pix --num_test 99999 --direction AtoB --output_nc 3 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 2
mkdir checkpoints/eliceiri_p2p_rot0-5_b2a
cp checkpoints/eliceiri_p2p_train_b2a/latest_net_* checkpoints/eliceiri_p2p_rot0-5_b2a
python test.py --dataroot ./datasets/eliceiri_patches_p2p/rot0-5 --name eliceiri_p2p_rot0-5_b2a --model pix2pix --num_test 99999 --direction BtoA --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 2

# level 2
mkdir checkpoints/eliceiri_p2p_rot5-10_a2b
cp checkpoints/eliceiri_p2p_train_a2b/latest_net_* checkpoints/eliceiri_p2p_rot5-10_a2b
python test.py --dataroot ./datasets/eliceiri_patches_p2p/rot5-10 --name eliceiri_p2p_rot5-10_a2b --model pix2pix --num_test 99999 --direction AtoB --output_nc 3 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 2
mkdir checkpoints/eliceiri_p2p_rot5-10_b2a
cp checkpoints/eliceiri_p2p_train_b2a/latest_net_* checkpoints/eliceiri_p2p_rot5-10_b2a
python test.py --dataroot ./datasets/eliceiri_patches_p2p/rot5-10 --name eliceiri_p2p_rot5-10_b2a --model pix2pix --num_test 99999 --direction BtoA --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 2

# level 3
mkdir checkpoints/eliceiri_p2p_rot10-15_a2b
cp checkpoints/eliceiri_p2p_train_a2b/latest_net_* checkpoints/eliceiri_p2p_rot10-15_a2b
python test.py --dataroot ./datasets/eliceiri_patches_p2p/rot10-15 --name eliceiri_p2p_rot10-15_a2b --model pix2pix --num_test 99999 --direction AtoB --output_nc 3 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 2
mkdir checkpoints/eliceiri_p2p_rot10-15_b2a
cp checkpoints/eliceiri_p2p_train_b2a/latest_net_* checkpoints/eliceiri_p2p_rot10-15_b2a
python test.py --dataroot ./datasets/eliceiri_patches_p2p/rot10-15 --name eliceiri_p2p_rot10-15_b2a --model pix2pix --num_test 99999 --direction BtoA --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 2

# level 4
mkdir checkpoints/eliceiri_p2p_rot15-20_a2b
cp checkpoints/eliceiri_p2p_train_a2b/latest_net_* checkpoints/eliceiri_p2p_rot15-20_a2b
python test.py --dataroot ./datasets/eliceiri_patches_p2p/rot15-20 --name eliceiri_p2p_rot15-20_a2b --model pix2pix --num_test 99999 --direction AtoB --output_nc 3 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 2
mkdir checkpoints/eliceiri_p2p_rot15-20_b2a
cp checkpoints/eliceiri_p2p_train_b2a/latest_net_* checkpoints/eliceiri_p2p_rot15-20_b2a
python test.py --dataroot ./datasets/eliceiri_patches_p2p/rot15-20 --name eliceiri_p2p_rot15-20_b2a --model pix2pix --num_test 99999 --direction BtoA --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 2





# cycleGAN

## Test 4-level data
### prepare
mkdir datasets/eliceiri_patches_cyc

mkdir datasets/eliceiri_patches_cyc/rot0-5
cp -r ../Datasets/Eliceiri_patches/patch_trans0-20_rot0-5/A/test/ ./datasets/eliceiri_patches_cyc/rot0-5/testA
cp -r ../Datasets/Eliceiri_patches/patch_trans0-20_rot0-5/B/test/ ./datasets/eliceiri_patches_cyc/rot0-5/testB

mkdir datasets/eliceiri_patches_cyc/rot5-10
cp -r ../Datasets/Eliceiri_patches/patch_trans20-40_rot5-10/A/test/ ./datasets/eliceiri_patches_cyc/rot5-10/testA
cp -r ../Datasets/Eliceiri_patches/patch_trans20-40_rot5-10/B/test/ ./datasets/eliceiri_patches_cyc/rot5-10/testB

mkdir datasets/eliceiri_patches_cyc/rot10-15
cp -r ../Datasets/Eliceiri_patches/patch_trans40-60_rot10-15/A/test/ ./datasets/eliceiri_patches_cyc/rot10-15/testA
cp -r ../Datasets/Eliceiri_patches/patch_trans40-60_rot10-15/B/test/ ./datasets/eliceiri_patches_cyc/rot10-15/testB

mkdir datasets/eliceiri_patches_cyc/rot15-20
cp -r ../Datasets/Eliceiri_patches/patch_trans60-80_rot15-20/A/test/ ./datasets/eliceiri_patches_cyc/rot15-20/testA
cp -r ../Datasets/Eliceiri_patches/patch_trans60-80_rot15-20/B/test/ ./datasets/eliceiri_patches_cyc/rot15-20/testB

### test
mkdir checkpoints/eliceiri_cyc_rot0-5
cp checkpoints/eliceiri_cyc_train/latest_net_* checkpoints/eliceiri_cyc_rot0-5
python test.py --dataroot ./datasets/eliceiri_patches_cyc/rot0-5/ --name eliceiri_cyc_rot0-5 --model cycle_gan --num_test 99999 --batch_size 1 --preprocess pad --divisor 256 --gpu_ids 2

mkdir checkpoints/eliceiri_cyc_rot5-10
cp checkpoints/eliceiri_cyc_train/latest_net_* checkpoints/eliceiri_cyc_rot5-10
python test.py --dataroot ./datasets/eliceiri_patches_cyc/rot5-10/ --name eliceiri_cyc_rot5-10 --model cycle_gan --num_test 99999 --batch_size 1 --preprocess pad --divisor 256 --gpu_ids 2

mkdir checkpoints/eliceiri_cyc_rot10-15
cp checkpoints/eliceiri_cyc_train/latest_net_* checkpoints/eliceiri_cyc_rot10-15
python test.py --dataroot ./datasets/eliceiri_patches_cyc/rot10-15/ --name eliceiri_cyc_rot10-15 --model cycle_gan --num_test 99999 --batch_size 1 --preprocess pad --divisor 256 --gpu_ids 2

mkdir checkpoints/eliceiri_cyc_rot15-20
cp checkpoints/eliceiri_cyc_train/latest_net_* checkpoints/eliceiri_cyc_rot15-20
python test.py --dataroot ./datasets/eliceiri_patches_cyc/rot15-20/ --name eliceiri_cyc_rot15-20 --model cycle_gan --num_test 99999 --batch_size 1 --preprocess pad --divisor 256 --gpu_ids 2


### unpad results
python ../utils/unpad_results.py -p ./results/eliceiri_p2p_rot0-5_a2b/test_latest/images
python ../utils/unpad_results.py -p ./results/eliceiri_p2p_rot0-5_b2a/test_latest/images
python ../utils/unpad_results.py -p ./results/eliceiri_p2p_rot5-10_a2b/test_latest/images
python ../utils/unpad_results.py -p ./results/eliceiri_p2p_rot5-10_b2a/test_latest/images
python ../utils/unpad_results.py -p ./results/eliceiri_p2p_rot10-15_a2b/test_latest/images
python ../utils/unpad_results.py -p ./results/eliceiri_p2p_rot10-15_b2a/test_latest/images
python ../utils/unpad_results.py -p ./results/eliceiri_p2p_rot15-20_a2b/test_latest/images
python ../utils/unpad_results.py -p ./results/eliceiri_p2p_rot15-20_b2a/test_latest/images

python ../utils/unpad_results.py -p ./results/eliceiri_cyc_rot0-5/test_latest/images
python ../utils/unpad_results.py -p ./results/eliceiri_cyc_rot5-10/test_latest/images
python ../utils/unpad_results.py -p ./results/eliceiri_cyc_rot10-15/test_latest/images
python ../utils/unpad_results.py -p ./results/eliceiri_cyc_rot15-20/test_latest/images

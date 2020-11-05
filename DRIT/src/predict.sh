# Eliceiri's data

## Test 4-level
### rot0-5
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_patches_cyc/rot0-5/ --resize_size 834 --crop_size 834 --a2b 1 --name eliceiri_drit_rot0-5_a2b --concat 0 --resume ../results/eliceiri_drit_train/01199.pth
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_patches_cyc/rot0-5/ --resize_size 834 --crop_size 834 --a2b 0 --name eliceiri_drit_rot0-5_b2a --concat 0 --resume ../results/eliceiri_drit_train/01199.pth
### rot5-10
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_patches_cyc/rot5-10/ --resize_size 834 --crop_size 834 --a2b 1 --name eliceiri_drit_rot5-10_a2b --concat 0 --resume ../results/eliceiri_drit_train/01199.pth
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_patches_cyc/rot5-10/ --resize_size 834 --crop_size 834 --a2b 0 --name eliceiri_drit_rot5-10_b2a --concat 0 --resume ../results/eliceiri_drit_train/01199.pth
### rot10-15
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_patches_cyc/rot10-15/ --resize_size 834 --crop_size 834 --a2b 1 --name eliceiri_drit_rot10-15_a2b --concat 0 --resume ../results/eliceiri_drit_train/01199.pth
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_patches_cyc/rot10-15/ --resize_size 834 --crop_size 834 --a2b 0 --name eliceiri_drit_rot10-15_b2a --concat 0 --resume ../results/eliceiri_drit_train/01199.pth
### rot15-20
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_patches_cyc/rot15-20/ --resize_size 834 --crop_size 834 --a2b 1 --name eliceiri_drit_rot15-20_a2b --concat 0 --resume ../results/eliceiri_drit_train/01199.pth
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_patches_cyc/rot15-20/ --resize_size 834 --crop_size 834 --a2b 0 --name eliceiri_drit_rot15-20_b2a --concat 0 --resume ../results/eliceiri_drit_train/01199.pth


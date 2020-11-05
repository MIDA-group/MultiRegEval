# For Eliceiri's data 

## prepare data for CurveAlign
cp ../Datasets/Eliceiri_patches/patch_trans0-20_rot0-5/B/test/*_T.tif eliceiri_4levels/HE/
rename -v '_T' '_l1' ./eliceiri_4levels/HE/*_T.tif
cp ../Datasets/Eliceiri_patches/patch_trans20-40_rot5-10/B/test/*_T.tif eliceiri_4levels/HE/
rename -v '_T' '_l2' ./eliceiri_4levels/HE/*_T.tif
cp ../Datasets/Eliceiri_patches/patch_trans40-60_rot10-15/B/test/*_T.tif eliceiri_4levels/HE/
rename -v '_T' '_l3' ./eliceiri_4levels/HE/*_T.tif
cp ../Datasets/Eliceiri_patches/patch_trans60-80_rot15-20/B/test/*_T.tif eliceiri_4levels/HE/
rename -v '_T' '_l4' ./eliceiri_4levels/HE/*_T.tif

cp ../Datasets/Eliceiri_patches/patch_trans0-20_rot0-5/A/test/*_R.tif eliceiri_4levels/SHG/
rename -v '_R' '_l1' ./eliceiri_4levels/SHG/*_R.tif
cp ../Datasets/Eliceiri_patches/patch_trans20-40_rot5-10/A/test/*_R.tif eliceiri_4levels/SHG/
rename -v '_R' '_l2' ./eliceiri_4levels/SHG/*_R.tif
cp ../Datasets/Eliceiri_patches/patch_trans40-60_rot10-15/A/test/*_R.tif eliceiri_4levels/SHG/
rename -v '_R' '_l3' ./eliceiri_4levels/SHG/*_R.tif
cp ../Datasets/Eliceiri_patches/patch_trans60-80_rot15-20/A/test/*_R.tif eliceiri_4levels/SHG/
rename -v '_R' '_l4' ./eliceiri_4levels/SHG/*_R.tif

# run
cd curvealign
matlab -nodesktop -nosplash -r CurveAlign




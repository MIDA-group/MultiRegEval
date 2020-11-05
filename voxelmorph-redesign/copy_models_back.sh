# command to run: 
# ./copy_models_back.sh

for modeldir in `ls -d models/*supervised_rot*_b2a/`; do
	scp MIDA:/data2/jiahao/Registration/voxelmorph-redesign/${modeldir}0500.h5 ./${modeldir}
done

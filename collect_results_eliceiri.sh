# command to run: 
# ./collect_results_eliceiri.sh
# renew p2p related results

dataroot="./Datasets/Eliceiri_patches"
method="aAMD"
# preprocess="nopre"
mode="b2a"

# for datadir in `ls ${dataroot} | grep patch_trans*`; do
for datadir in `ls ${dataroot} | grep tlevel*`; do
	mkdir -p ./logs/eliceiri_${datadir}
	for gan in p2p_A p2p_B; do
		for preprocess in nopre hiseq; do
			nohup python evaluate.py -d ${dataroot}/${datadir}/ -m ${method} -g ${gan} --mode ${mode} --pre ${preprocess} >./logs/eliceiri_${datadir}/${method}${gan}_${mode}_${preprocess}.out 2>&1 &
		done
	done
done

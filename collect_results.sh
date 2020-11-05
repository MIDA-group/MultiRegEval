# command to run: 
# ./collect_results.sh

dataroot="./Datasets/Eliceiri_patches"
method="aAMD"
preprocess="nopre"
mode="b2a"

for datadir in `ls ${dataroot} | grep patch_trans*`; do
	mkdir -p ./logs/${datadir}
	for gan in p2p_A p2p_B cyc_A cyc_B drit_A drit_B; do
		nohup python evaluate.py -d ${dataroot}/${datadir}/ -m ${method} -g ${gan} --mode ${mode} --pre ${preprocess} >logs/${datadir}/${method}${gan}_${mode}_${preprocess}.out 2>&1 &
	done
done

wait

preprocess="hiseq"

for datadir in `ls ${dataroot} | grep patch_trans*`; do
	mkdir -p ./logs/${datadir}
	for gan in p2p_A p2p_B cyc_A cyc_B drit_A drit_B; do
		nohup python evaluate.py -d ${dataroot}/${datadir}/ -m ${method} -g ${gan} --mode ${mode} --pre ${preprocess} >logs/${datadir}/${method}${gan}_${mode}_${preprocess}.out 2>&1 &
	done
done

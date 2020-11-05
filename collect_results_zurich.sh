# command to run: 
# ./collect_results_zurich.sh

dataroot="./Datasets/Zurich_patches/fold1"
method="aAMD"
preprocess="nopre"
mode="b2a"

# for datadir in `ls ${dataroot} | grep patch_trans*`; do
for datadir in `ls ${dataroot} | grep tlevel*`; do
	mkdir -p ./logs/zurich_fold1_${datadir}
	for gan in p2p_A p2p_B cyc_A cyc_B drit_A drit_B; do
		nohup python evaluate.py -d ${dataroot}/${datadir}/ -m ${method} -g ${gan} --mode ${mode} --pre ${preprocess} >./logs/zurich_fold1_${datadir}/${method}${gan}_${mode}_${preprocess}.out 2>&1 &
	done
done

wait

preprocess="hiseq"

# for datadir in `ls ${dataroot} | grep patch_trans*`; do
for datadir in `ls ${dataroot} | grep tlevel*`; do
	mkdir -p ./logs/zurich_fold1_${datadir}
	for gan in p2p_A p2p_B cyc_A cyc_B drit_A drit_B; do
		nohup python evaluate.py -d ${dataroot}/${datadir}/ -m ${method} -g ${gan} --mode ${mode} --pre ${preprocess} >./logs/zurich_fold1_${datadir}/${method}${gan}_${mode}_${preprocess}.out 2>&1 &
	done
done

wait

for datadir in `ls ${dataroot} | grep tlevel*`; do
	for mode in a2a b2b b2a; do
		for preprocess in nopre hiseq; do
			nohup python evaluate.py -d ${dataroot}/${datadir}/ -m ${method} --mode ${mode} --pre ${preprocess} >./logs/zurich_fold1_${datadir}/${method}_${mode}_${preprocess}.out 2>&1 &
		done
	done
done
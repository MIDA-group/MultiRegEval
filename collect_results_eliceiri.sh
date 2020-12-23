# command to run: 
# nohup ./collect_results_eliceiri.sh {method} &

dataroot="./Datasets/Eliceiri_patches"
# method="aAMD"
method=$1
preprocess="nopre"
mode="b2a"

# for datadir in `ls ${dataroot} | grep patch_trans*`; do
for datadir in `ls ${dataroot} | grep tlevel*`; do
	mkdir -p ./logs/eliceiri_${datadir}
	for gan in p2p_A p2p_B cyc_A cyc_B drit_A drit_B star_A star_B comir; do
		nohup python evaluate.py -d ${dataroot}/${datadir}/ -m ${method} -g ${gan} --mode ${mode} --pre ${preprocess} >./logs/eliceiri_${datadir}/${method}${gan}_${mode}_${preprocess}.out 2>&1 &
	done
done

wait

preprocess="hiseq"

# for datadir in `ls ${dataroot} | grep patch_trans*`; do
for datadir in `ls ${dataroot} | grep tlevel*`; do
	mkdir -p ./logs/eliceiri_${datadir}
	for gan in p2p_A p2p_B cyc_A cyc_B drit_A drit_B star_A star_B comir; do
		nohup python evaluate.py -d ${dataroot}/${datadir}/ -m ${method} -g ${gan} --mode ${mode} --pre ${preprocess} >./logs/eliceiri_${datadir}/${method}${gan}_${mode}_${preprocess}.out 2>&1 &
	done
done

wait

for datadir in `ls ${dataroot} | grep tlevel*`; do
	for mode in a2a b2b b2a; do
		for preprocess in nopre hiseq; do
			nohup python evaluate.py -d ${dataroot}/${datadir}/ -m ${method} --mode ${mode} --pre ${preprocess} >./logs/eliceiri_${datadir}/${method}_${mode}_${preprocess}.out 2>&1 &
		done
	done
done

wait
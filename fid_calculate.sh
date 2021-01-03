# collect FID results
# Run: nohup ./fid_calculate.sh &

# mida2
CUDA_VISIBLE_DEVICES=0 nohup python fid_calculate.py -d Eliceiri &

for fold in {1..3}; do
	CUDA_VISIBLE_DEVICES=1 nohup python fid_calculate.py -d Zurich -f ${fold} &
	CUDA_VISIBLE_DEVICES=2 nohup python fid_calculate.py -d Balvan -f ${fold} &
	wait
done


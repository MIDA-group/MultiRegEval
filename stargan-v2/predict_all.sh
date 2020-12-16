# predict all datasets
# Run: nohup ./predict_all.sh &

./predict_eliceiri.sh > ./checkpoints/predict_eliceiri.out 2>&1
./predict_balvan.sh > ./checkpoints/predict_balvan.out 2>&1
./predict_zurich.sh > ./checkpoints/predict_zurich.out 2>&1

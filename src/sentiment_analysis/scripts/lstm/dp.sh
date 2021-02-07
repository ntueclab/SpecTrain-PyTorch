MASTER_ADDR=localhost

NUM_GPU=2
LR=0.1

python main_with_runtime.py --module models.residual_lstm.gpus=4_straight -b 256 --data_dir ./data --config_path models/residual_lstm/gpus=4_straight/dp_conf.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs 20 --lr $LR --log_dir logs/dp_${NUM_GPU}_${LR} --lr_policy polynomial --rank 0 --local_rank 0 --world_size $NUM_GPU &
python main_with_runtime.py --module models.residual_lstm.gpus=4_straight -b 256 --data_dir ./data --config_path models/residual_lstm/gpus=4_straight/dp_conf.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs 20 --lr $LR --log_dir logs/dp_${NUM_GPU}_${LR} --lr_policy polynomial --rank 1 --local_rank 1 --world_size $NUM_GPU  
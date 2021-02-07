MASTER_ADDR=localhost

NUM_GPU=4
PORT=32904
EPOCHS=40
LR=0.09

python main_with_runtime.py --module models.residual_lstm.gpus=4_straight -b 256 --data_dir ./data --config_path models/residual_lstm/gpus=4_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs $EPOCHS --lr $LR --log_dir logs/mp_${NUM_GPU}_${LR} --lr_policy polynomial --rank 0 --local_rank 0 --world_size $NUM_GPU --spectrain &
python main_with_runtime.py --module models.residual_lstm.gpus=4_straight -b 256 --data_dir ./data --config_path models/residual_lstm/gpus=4_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs $EPOCHS --lr $LR --log_dir logs/mp_${NUM_GPU}_${LR} --lr_policy polynomial --rank 1 --local_rank 1 --world_size $NUM_GPU --spectrain &
python main_with_runtime.py --module models.residual_lstm.gpus=4_straight -b 256 --data_dir ./data --config_path models/residual_lstm/gpus=4_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs $EPOCHS --lr $LR --log_dir logs/mp_${NUM_GPU}_${LR} --lr_policy polynomial --rank 2 --local_rank 2 --world_size $NUM_GPU --spectrain &
python main_with_runtime.py --module models.residual_lstm.gpus=4_straight -b 256 --data_dir ./data --config_path models/residual_lstm/gpus=4_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs $EPOCHS --lr $LR --log_dir logs/mp_${NUM_GPU}_${LR} --lr_policy polynomial --rank 3 --local_rank 3 --world_size $NUM_GPU --spectrain 
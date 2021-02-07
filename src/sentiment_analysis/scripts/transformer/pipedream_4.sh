MASTER_ADDR=localhost

NUM_GPU=4
EPOCHS=50
LR=0.08

python main_with_runtime.py --module models.transformer.gpus=4_straight -b 256 --data_dir ./data --config_path models/transformer/gpus=4_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs $EPOCHS --lr $LR --log_dir logs/mp_${NUM_GPU}_${LR} --lr_policy polynomial --rank 0 --local_rank 0 --world_size $NUM_GPU &
python main_with_runtime.py --module models.transformer.gpus=4_straight -b 256 --data_dir ./data --config_path models/transformer/gpus=4_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs $EPOCHS --lr $LR --log_dir logs/mp_${NUM_GPU}_${LR} --lr_policy polynomial --rank 1 --local_rank 1 --world_size $NUM_GPU &
python main_with_runtime.py --module models.transformer.gpus=4_straight -b 256 --data_dir ./data --config_path models/transformer/gpus=4_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs $EPOCHS --lr $LR --log_dir logs/mp_${NUM_GPU}_${LR} --lr_policy polynomial --rank 2 --local_rank 2 --world_size $NUM_GPU &
python main_with_runtime.py --module models.transformer.gpus=4_straight -b 256 --data_dir ./data --config_path models/transformer/gpus=4_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs $EPOCHS --lr $LR --log_dir logs/mp_${NUM_GPU}_${LR} --lr_policy polynomial --rank 3 --local_rank 3 --world_size $NUM_GPU 
MASTER_ADDR=localhost

NUM_GPU=2
LR=0.1
EPOCHS=20

python main_with_runtime_chc.py --module models.transformer.gpus=4_straight -b 256 --data_dir ./data --config_path models/transformer/gpus=4_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs $EPOCHS --lr $LR --log_dir logs/mp_${NUM_GPU}_${LR} --lr_policy polynomial --rank 0 --local_rank 0 --world_size $NUM_GPU --spectrain &
python main_with_runtime_chc.py --module models.transformer.gpus=4_straight -b 256 --data_dir ./data --config_path models/transformer/gpus=4_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs $EPOCHS --lr $LR --log_dir logs/mp_${NUM_GPU}_${LR} --lr_policy polynomial --rank 1 --local_rank 1 --world_size $NUM_GPU --spectrain 
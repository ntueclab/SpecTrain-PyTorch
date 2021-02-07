NUM_GPU=4
MASTER_ADDR=localhost
DECAY=0.0005
EPOCH=50
LR=0.0075

python main_with_runtime.py --module models.googlenet.gpus=2 -b 128 --data_dir ./data --config_path models/googlenet/gpus=2/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs ${EPOCH} --lr $LR --weight-decay $DECAY --log_dir logs_resnet/mp_${NUM_GPU}_LR${LR}_decay${DECAY}_${EPOCH} --lr_policy polynomial --rank 0 --local_rank 0 --world_size 4 &
python main_with_runtime.py --module models.googlenet.gpus=2 -b 128 --data_dir ./data --config_path models/googlenet/gpus=2/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs ${EPOCH} --lr $LR --weight-decay $DECAY --log_dir logs_resnet/mp_${NUM_GPU}_LR${LR}_decay${DECAY}_${EPOCH} --lr_policy polynomial --rank 1 --local_rank 1 --world_size 4 &
python main_with_runtime.py --module models.googlenet.gpus=2 -b 128 --data_dir ./data --config_path models/googlenet/gpus=2/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs ${EPOCH} --lr $LR --weight-decay $DECAY --log_dir logs_resnet/mp_${NUM_GPU}_LR${LR}_decay${DECAY}_${EPOCH} --lr_policy polynomial --rank 2 --local_rank 2 --world_size 4 &
python main_with_runtime.py --module models.googlenet.gpus=2 -b 128 --data_dir ./data --config_path models/googlenet/gpus=2/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs ${EPOCH} --lr $LR --weight-decay $DECAY --log_dir logs_resnet/mp_${NUM_GPU}_LR${LR}_decay${DECAY}_${EPOCH} --lr_policy polynomial --rank 3 --local_rank 3 --world_size 4
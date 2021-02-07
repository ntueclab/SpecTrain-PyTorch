NUM_GPU=2
MASTER_ADDR=localhost
DECAY=0.01
EPOCH=100
LR=0.0015

python main_with_runtime_chc.py --module models.resnet50.gpus=2 -b 128 --data_dir ./data --config_path models/resnet50/gpus=2/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs ${EPOCH} --lr $LR --weight-decay $DECAY --log_dir logs_resnet/spectrain_${NUM_GPU}_LR${LR}_decay${DECAY}_${EPOCH} --lr_policy polynomial --rank 0 --local_rank 0 --world_size 2 --spectrain &
python main_with_runtime_chc.py --module models.resnet50.gpus=2 -b 128 --data_dir ./data --config_path models/resnet50/gpus=2/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs ${EPOCH} --lr $LR --weight-decay $DECAY --log_dir logs_resnet/spectrain_${NUM_GPU}_LR${LR}_decay${DECAY}_${EPOCH} --lr_policy polynomial --rank 1 --local_rank 1 --world_size 2 --spectrain 
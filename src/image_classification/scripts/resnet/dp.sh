NUM_GPU=2
MASTER_ADDR=localhost
DECAY=0.01
EPOCH=100

LR=0.01
PREFIX=dp
python main_with_runtime.py --module models.resnet50.gpus=2 -b 128 --data_dir ./data --config_path models/resnet50/gpus=2/dp_conf.json --distributed_backend nccl --master_addr $MASTER_ADDR --epochs ${EPOCH} --lr $LR --weight-decay $DECAY --log_dir logs_resnet/${PREFIX}_${NUM_GPU}_LR${LR}_decay${DECAY}_${EPOCH} --lr_policy polynomial --rank 0 --local_rank 0 --world_size 2 &
python main_with_runtime.py --module models.resnet50.gpus=2 -b 128 --data_dir ./data --config_path models/resnet50/gpus=2/dp_conf.json --distributed_backend nccl --master_addr $MASTER_ADDR --epochs ${EPOCH} --lr $LR --weight-decay $DECAY --log_dir logs_resnet/${PREFIX}_${NUM_GPU}_LR${LR}_decay${DECAY}_${EPOCH} --lr_policy polynomial --rank 1 --local_rank 1 --world_size 2
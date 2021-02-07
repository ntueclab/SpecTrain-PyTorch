MASTER_ADDR=localhost

NUM_GPU=4
LR=0.02

python main_with_runtime_chc.py --module models.vgg16.gpus=16_straight -b 128 --data_dir ./data --config_path models/vgg16/gpus=16_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs 100 --lr $LR --weight-decay 0.0005 --log_dir logs/spectrain_${NUM_GPU}_lr${LR} --lr_policy polynomial --rank 0 --local_rank 0 --world_size 4 --spectrain &
python main_with_runtime_chc.py --module models.vgg16.gpus=16_straight -b 128 --data_dir ./data --config_path models/vgg16/gpus=16_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs 100 --lr $LR --weight-decay 0.0005 --log_dir logs/spectrain_${NUM_GPU}_lr${LR} --lr_policy polynomial --rank 1 --local_rank 1 --world_size 4 --spectrain &
python main_with_runtime_chc.py --module models.vgg16.gpus=16_straight -b 128 --data_dir ./data --config_path models/vgg16/gpus=16_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs 100 --lr $LR --weight-decay 0.0005 --log_dir logs/spectrain_${NUM_GPU}_lr${LR} --lr_policy polynomial --rank 2 --local_rank 2 --world_size 4 --spectrain &
python main_with_runtime_chc.py --module models.vgg16.gpus=16_straight -b 128 --data_dir ./data --config_path models/vgg16/gpus=16_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs 100 --lr $LR --weight-decay 0.0005 --log_dir logs/spectrain_${NUM_GPU}_lr${LR} --lr_policy polynomial --rank 3 --local_rank 3 --world_size 4 --spectrain
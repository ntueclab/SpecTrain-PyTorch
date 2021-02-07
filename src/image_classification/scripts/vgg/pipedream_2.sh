MASTER_ADDR=localhost

NUM_GPU=2
LR=0.01

python main_with_runtime.py --module models.vgg16.gpus=16_straight -b 128 --data_dir ./data --config_path models/vgg16/gpus=16_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs 100 --lr $LR --weight-decay 0.0005 --log_dir logs/mp_${NUM_GPU}_${LR}_pipedream --lr_policy polynomial --rank 0 --local_rank 0 --world_size 2 &
python main_with_runtime.py --module models.vgg16.gpus=16_straight -b 128 --data_dir ./data --config_path models/vgg16/gpus=16_straight/mp_conf_${NUM_GPU}gpu.json --distributed_backend gloo --master_addr $MASTER_ADDR --epochs 100 --lr $LR --weight-decay 0.0005 --log_dir logs/mp_${NUM_GPU}_${LR}_pipedream --lr_policy polynomial --rank 1 --local_rank 1 --world_size 2 
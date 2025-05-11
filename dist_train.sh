# train
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1  --master_port 51119  --use_env ./Adafreq/train.py --config_file ./Adafreq/configs/ATRW/vit_transreid_stride.yml MODEL.DIST_TRAIN True
# test
python ./Adafreq/test.py --config_file ./Adafreq/configs/ATRW/vit_transreid_stride.yml MODEL.DEVICE_ID "('2')"

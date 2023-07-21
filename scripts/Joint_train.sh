coco_path=$1
python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py  -c config/ESTS/ESTS_5scale_joint_train.py --coco_path $coco_path --output_dir logs/ESTS/R50-MS5_Joint_train \
        --train_dataset totaltext_train:ic13_train:ic15_train:mlt_train \
        --val_dataset totaltext_val \
        --pretrain_model_path logs/ESTS/R50-MS5_Pretrain/checkpoint0024.pth \
        --options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
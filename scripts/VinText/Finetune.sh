coco_path=$1
python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py  -c config/ESTS/ESTS_5scale_vintext_finetune.py --coco_path $coco_path --output_dir logs/ESTS/R50-MS5_vintext_finetune \
        --train_dataset vintext:vintext_val \
        --val_dataset vintext_test \
        --pretrain_model_path logs/ESTS/R50-MS5_Joint_train/checkpoint0060.pth \
        --options dn_scalar=100 embed_init_tgt=TRUE \
		dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
		dn_box_noise_scale=1.0
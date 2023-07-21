coco_path=$1
python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py  -c config/ESTS/ESTS_4scale_mlt_pretrain.py --coco_path $coco_path --output_dir logs/ESTS/R50-MS4_MLT_Pretrain \
        --train_dataset totaltext_train:ic13_train:ic15_train:mlt_train:syntext1_train:syntext2_train \
        --val_dataset totaltext_val \
        --options dn_scalar=100 embed_init_tgt=TRUE \
		dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
		dn_box_noise_scale=1.0
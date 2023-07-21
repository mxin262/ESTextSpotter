coco_path=$1
python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py  -c config/ESTS/ESTS_4scale_chn_pretrain.py --coco_path $coco_path --output_dir logs/ESTS/R50-MS4_Chn_Pretrain \
        --train_dataset art:lsvt:rects:rects_val:chnsyntext \
        --val_dataset rects_test \
        --options dn_scalar=100 embed_init_tgt=TRUE \
		dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
		dn_box_noise_scale=1.0
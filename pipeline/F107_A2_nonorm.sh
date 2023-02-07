python 00_start_pipeline.py /scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A2_em.n5  "raw_zero_bg" /scratch/buglakova/data/cryofib/segm_fibsem/F107/multicut_pipeline/F107_A2_em_pipeline_nonorm.n5

python 01_get_fg_mask.py /scratch/buglakova/data/cryofib/segm_fibsem/F107/multicut_pipeline/F107_A2_em_pipeline_nonorm.n5

python 02_quantile_norm.py /scratch/buglakova/data/cryofib/segm_fibsem/F107/multicut_pipeline/F107_A2_em_pipeline_nonorm.n5

export CUDA_VISIBLE_DEVICES=7
python 03_predict_boundaries.py /scratch/buglakova/data/cryofib/segm_fibsem/F107/multicut_pipeline/F107_A2_em_pipeline_nonorm.n5 /g/kreshuk/buglakova/projects/fibsem_segm/unet_F107_A1_full/experiments/exp_08_dice_norm_64features_16x512x512/modelzoo/xp_08_dice_norm_64features_16x512x512/exp_08_dice_norm_64features_16x512x512.zip

# python 04_3D_watershed.py /scratch/buglakova/data/cryofib/segm_fibsem/F107/multicut_pipeline/F107_A2_em_pipeline.n5

# python 05_multicut.py /scratch/buglakova/data/cryofib/segm_fibsem/F107/multicut_pipeline/F107_A2_em_pipeline.n5 --beta 0.5

# python 07_postprocess_extra_size.py /scratch/buglakova/data/cryofib/segm_fibsem/F107/multicut_pipeline/F107_A2_em_pipeline.n5
python 00_start_pipeline.py /scratch/buglakova/data/cryofib/segm_fibsem/F059/F059_A1_em.n5  "raw" /scratch/buglakova/data/cryofib/segm_fibsem/F059/multicut_pipeline/F059_A1_em_pipeline.n5

python 01_get_fg_mask.py /scratch/buglakova/data/cryofib/segm_fibsem/F059/multicut_pipeline/F059_A1_em_pipeline.n5

python 02_quantile_norm.py /scratch/buglakova/data/cryofib/segm_fibsem/F059/multicut_pipeline/F059_A1_em_pipeline.n5

export export CUDA_VISIBLE_DEVICES=7
python 03_predict_boundaries.py /scratch/buglakova/data/cryofib/segm_fibsem/F059/multicut_pipeline/F059_A1_em_pipeline.n5 /g/kreshuk/buglakova/projects/fibsem_segm/unet_F107_A1_full/experiments/exp_10_dice_F056_A1/modelzoo/exp_10_dice_F056_A1/exp_10_dice_F056_A1.zip

python 04_3D_watershed.py /scratch/buglakova/data/cryofib/segm_fibsem/F059/multicut_pipeline/F059_A1_em_pipeline.n5

python 05_multicut.py /scratch/buglakova/data/cryofib/segm_fibsem/F059/multicut_pipeline/F059_A1_em_pipeline.n5 --beta 0.5

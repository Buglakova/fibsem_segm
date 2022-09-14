python convert_to_n5_F107.py

conda activate biomodel_env
export BIOIMAGEIO_CACHE_PATH="/g/kreshuk/buglakova/biomodel_zoo_tmp/"
CUDA_VISIBLE_DEVICES=4 python adrian_unet_predict.py /g/kreshuk/buglakova/data/cryofib/registration_fluo/F107/F107_fluo.n5/ "raw" /g/kreshuk/buglakova/data/cryofib/registration_fluo/F107/F107_fluo.n5/ "segmentation/adrian_unet"


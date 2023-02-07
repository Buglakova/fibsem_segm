python /g/kreshuk/buglakova/projects/fibsem_segm/multicut/train_multicut/watershed_3D.py "/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A2_em.n5" "input/raw_norm" \
 "/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A2_em_thin_boundaries_predictions.n5" "dilated_boundary_predictions/dice_16x512x512/boundaries" \
 "/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A2_em_thin_boundaries_predictions.n5" "dilated_boundary_predictions/dice_16x512x512/extra" \
  "/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A2_em_thin_boundaries_predictions.n5" "3D_watershed" \
  --n_threads 8
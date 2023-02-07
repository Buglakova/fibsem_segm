import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cryofib.n5_utils import read_volume, write_volume, get_attrs
from cryofib.preprocess_utils import percentile_norm
from cryofib.vis import plot_three_slices
from cryofib.prediction_utils import predict_network



def main():
 
    ## Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""Predict boundaries with a trained model.
        """
    )

    parser.add_argument("pipeline_n5", type=str, help="Path of the input n5")
    parser.add_argument("model_path", type=str, help="Path to modelzoo model")
    
    args = parser.parse_args()
    
    n5_path = Path(args.pipeline_n5)
    
    log_dir = n5_path.parent / (n5_path.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = n5_path.parent / (n5_path.stem + "_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(log_dir / "03_predict_boundaries.log", mode="w"),
                            logging.StreamHandler(sys.stdout)],
                    datefmt='%Y-%m-%d %H:%M:%S')
    
    # Read data
    img = read_volume(args.pipeline_n5, "input/raw_norm")
    assert img is not None, "Key doesn't exist"
    
    logging.info(f"Start prediction")
    # Normalize
    pred = predict_network(img, args.model_path, args.pipeline_n5, "prediction")
    logging.info(f"End prediction")
    # Plots
    for name, chan in zip(["fg", "boundaries", "extra", "bg"], pred):
        plot_three_slices(chan, plots_dir / f"pred_{name}.png", cmap="Greys_r")
    
    
if __name__ == "__main__":
    main()
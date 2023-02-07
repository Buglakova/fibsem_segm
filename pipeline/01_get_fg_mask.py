import argparse
from pathlib import Path
import numpy as np

from cryofib.n5_utils import read_volume, write_volume, get_attrs
from cryofib.preprocess_utils import get_fg_mask
from cryofib.vis import plot_three_slices


def main():
    ## Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""Find a foreground mask. Background is the biggest connected component of zero pixels per z slice.
        """
    )

    parser.add_argument("pipeline_n5", type=str, help="Path of the input n5")
    parser.add_argument("--bg_const", type=float, default=0, help="Optional key for the fg mask")
    parser.add_argument("--output_n5_key", type=str, default="input/fg_mask", help="Optional key for the fg mask")
    
    args = parser.parse_args()

    n5_path = Path(args.pipeline_n5)
    plots_dir = n5_path.parent / (n5_path.stem + "_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    raw = read_volume(args.pipeline_n5, "input/raw")
    assert raw is not None, "Key doesn't exist"
    
    fg_mask = get_fg_mask(raw, bg_const=args.bg_const)
    print(fg_mask)
    
    plot_three_slices(fg_mask, plots_dir / "fg_mask.png", cmap="Greys_r")
    
    
    attrs = {"description": "Foreground mask. Background is the biggest connected component of zero pixels per z slice."}
    write_volume(args.pipeline_n5, fg_mask.astype(np.uint8), args.output_n5_key, attrs=attrs)
    
    
if __name__ == "__main__":
    main()
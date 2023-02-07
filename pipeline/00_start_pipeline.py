import argparse
import numpy as np

from cryofib.n5_utils import read_volume, write_volume, get_attrs


def main():
    ## Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""Start running a pipeline by copying raw data to a new n5 file.
        """
    )

    parser.add_argument("raw_n5", type=str, help="Path of the input n5")
    parser.add_argument("raw_n5_key", type=str, help="Key of the dataset in the input n5")

    parser.add_argument("output_n5", type=str, help="Path of the output n5")
    parser.add_argument("--output_n5_key", type=str, default="input/raw", help="Optional key in the output file to write raw")
    
    args = parser.parse_args()

    raw = read_volume(args.raw_n5, args.raw_n5_key)
    
    assert raw is not None, "Key doesn't exist"
    
    raw_attrs = get_attrs(args.raw_n5, args.raw_n5_key)
    write_volume(args.output_n5, raw, args.output_n5_key, attrs=raw_attrs)
    
    
if __name__ == "__main__":
    main()
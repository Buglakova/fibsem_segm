import os

import h5py
from torch_em.util import export_bioimageio_model, export_parser_helper, get_default_citations
import z5py
from cryofib.n5_utils import read_volume
import numpy as np


def export_model(checkpoint, output):
    raw_path = "/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5"
    raw_f = z5py.File(raw_path, "r")
    roi = np.s_[182:247, :, :]
    raw_data = read_volume(raw_f, "raw", roi)
    z_sh, y_sh, x_sh = (64, 256, 256)
    print(raw_data.shape)
    input_data = raw_data[:z_sh, :y_sh, :x_sh]
    print(input_data.shape)
    cite = get_default_citations(model="UNet3D", model_output="boundaries")
    export_bioimageio_model(
        checkpoint, output, input_data,
        name=os.path.basename(checkpoint),
        authors=[{"name": "Elena Buglakova; @Buglakova"}],
        tags=["unet"],
        license="CC-BY-4.0",
        documentation="UNet3D for membranes trained on organoids EM data with Dice Loss.",
        git_repo="https://github.com/Buglakova/fibsem_segm.git",
        cite=cite,
        input_optional_parameters=False,
    )


def main():
    parser = export_parser_helper()
    args = parser.parse_args()
    export_model(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
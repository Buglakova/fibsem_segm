import os

import h5py
from torch_em.util import export_bioimageio_model, export_parser_helper, get_default_citations
import z5py
from cryofib.data_loaders import load_platynereis_memb_ds


def export_model(checkpoint, output):
    raw_list, label_list = load_platynereis_memb_ds()
    z_sh, y_sh, x_sh = (64, 128, 128)
    print(raw_list[0].shape)
    input_data = raw_list[0][:z_sh, :y_sh, :x_sh]
    print(input_data.shape)
    cite = get_default_citations(model="UNet3D", model_output="boundaries")
    export_bioimageio_model(
        checkpoint, output, input_data,
        name=os.path.basename(checkpoint),
        authors=[{"name": "Elena Buglakova; @Buglakova"}],
        tags=["unet"],
        license="CC-BY-4.0",
        documentation="UNet3D for membranes trained on platynereis EM data as a test with masked Dice Loss.",
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
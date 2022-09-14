def save_modelzoo():
    # Save model in bioimage io format

    export_folder = "./bio-model"

    # Whether to convert the model weights to additional formats.
    # Currently, torchscript and onnx are support it and this will enable running the model
    # in more software tools.
    additional_weight_formats = None
    # additional_weight_formats = ["torchscript"]

    doc = """# Default model
    Try to run the training with default parameters just to see that it's training and prediction is working.
    """

    import torch_em.util

    for_dij = additional_weight_formats is not None and "torchscript" in additional_weight_formats

    training_data = None

    pred_str = "out_boundary_extra_fg"

    default_doc = f"""#{experiment_name}

    This model was trained with [the torch_em 3d UNet notebook](https://github.com/constantinpape/torch-em/blob/main/experiments/3D-UNet-Training.ipynb).
    """
    if pred_str:
        default_doc += f"It predicts {pred_str}.\n"

    training_summary = torch_em.util.get_training_summary(trainer, to_md=True, lr=learning_rate)
    default_doc += f"""## Training Schedule

    {training_summary}
    """

    if doc is None:
        doc = default_doc

    torch_em.util.export_bioimageio_model(
        trainer, export_folder, input_optional_parameters=True,
        for_deepimagej=for_dij, training_data=training_data, documentation=doc
    )
    torch_em.util.add_weight_formats(export_folder, additional_weight_formats)

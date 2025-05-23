from lightning.pytorch.cli import LightningCLI


class Py4castLightningCLI(LightningCLI):
    """
    CLI - Command Line Interface from lightning
    Args:
        A model which inherits from LightningModule
        A datamodule which inherits from LightningDataModule
        save_config_kwargs define if checkpoint should be stored even if one is already
        present in the folder, useful for development.
    """

    def __init__(self, model_class, datamodule_class, *args, **kwargs):
        super().__init__(
            model_class,
            datamodule_class,
            *args,
            **kwargs,
        )

    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.dataset_name",
            "model.dataset_name",
        )
        parser.link_arguments(
            "data.batch_size",
            "model.batch_size",
        )
        parser.link_arguments(
            "data.num_input_steps",
            "model.num_input_steps",
        )
        parser.link_arguments(
            "data.num_pred_steps_train",
            "model.num_pred_steps_train",
        )
        parser.link_arguments(
            "data.num_pred_steps_val_test",
            "model.num_pred_steps_val_test",
        )
        parser.link_arguments(
            "data.dataset_conf",
            "model.dataset_conf",
        )
        parser.link_arguments(
            "data.train_dataset_info",
            "model.dataset_info",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.infer_ds",
            "model.infer_ds",
            apply_on="instantiate",
        )

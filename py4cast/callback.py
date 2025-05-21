from typing import Any, Literal, Optional, Union
from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_warn, rank_zero_info


class FineTuningScheduler(Callback):
    """
    Callback to implement curriculum learning for autoregressive steps.

    During pretraining, the number of autoregressive steps is fixed at 1.
    During fine-tuning, autoregressive steps increase by `steps_increment`
    every `update_frequency` training steps.

    When increasing autoregressive steps, this callback updates the model's
    autoregressive_steps attribute and modifies the trainer's limit_train_batches.

    Args:
    max_autoregressive_steps: Maximum number of autoregressive steps
    steps_increment: Number of steps to increase by at each update
    update_frequency: How often to increase steps, in training steps
    init_autoregressive_steps: Initial number of autoregressive steps
    pretraining_mode: If True, keeps autoregressive_steps=1
    pretraining_steps: Number of steps allocated for pretraining phase
    finetuning_steps: Number of steps allocated for finetuning phase
    batches_per_step_increase: Number of batches to process after each step increase
                                (if None, uses all available batches)
    """

    def __init__(
        self,
        logging_interval: Optional[Literal["epoch", "step"]] = "None",
        max_autoregressive_steps: int = 6,
        steps_increment: int = 1,
        update_frequency: int = 1000,
        pretraining_steps: int = 0,
        batches_per_step_increase: int = None,
        init_autoregressive_steps: int = 1,  # Added for clarity
    ) -> None:

        if logging_interval not in ("None", "epoch", "step"):
            raise MisconfigurationException(
                f"Invalid logging_interval: {logging_interval}. "
                "Must be 'None', 'epoch', or 'step'."
            )
        self.logging_interval = logging_interval
        if max_autoregressive_steps < 1:
            raise MisconfigurationException(
                f"max_autoregressive_steps must be >= 1, got {max_autoregressive_steps}."
            )
        if steps_increment < 1:
            raise MisconfigurationException(
                f"steps_increment must be >= 1, got {steps_increment}."
            )
        if update_frequency < 1:
            raise MisconfigurationException(
                f"update_frequency must be >= 1, got {update_frequency}."
            )
        if pretraining_steps < 0:
            raise MisconfigurationException(
                f"pretraining_steps must be >= 0, got {pretraining_steps}."
            )

        if init_autoregressive_steps < 1:
            raise MisconfigurationException(
                f"init_autoregressive_steps must be >= 1, got {init_autoregressive_steps}."
            )

        self.max_autoregressive_steps = max_autoregressive_steps
        self.steps_increment = steps_increment
        self.update_frequency = update_frequency
        self.pretraining_steps = pretraining_steps
        self.batches_per_step_increase = batches_per_step_increase
        self.init_autoregressive_steps = init_autoregressive_steps

        self.current_autoregressive_steps = self.init_autoregressive_steps
        self.steps_completed_in_phase = 0

        self.original_limit_train_batches: Optional[Union[int, float]] = None
        self.original_limit_val_batches: Optional[Union[int, float]] = None
        self._reset_train_batch_limit_on_epoch_start = False
        self._reset_val_batch_limit_on_epoch_start = False
        self._last_logged_step = -1
        self.finetuning_steps_calculated: Optional[int] = (
            None  # For validate_total_steps
        )

    # @override
    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        Verify that the model has the required attributes.
        """
        if not trainer.loggers and self.logging_interval != "None":
            rank_zero_warn(
                "FineTuningScheduler: No logger found, but logging_interval is not 'None'. Metrics will not be logged."
            )

        if not hasattr(pl_module, "num_pred_steps_train"):
            raise MisconfigurationException(
                "The model must have a `num_pred_steps_train` attribute for FineTuningScheduler to work."
            )
        if not hasattr(trainer, "datamodule"):
            raise MisconfigurationException(
                "The model must have a `datamodule` attribute."
            )
        if not hasattr(trainer.datamodule, "num_pred_steps_train"):
            raise MisconfigurationException(
                "datamodule.num_pred_steps_train must have a `num_pred_steps_train` attribute."
            )
        if not hasattr(trainer.datamodule, "num_pred_steps_val_test"):
            rank_zero_warn(
                "FineTuningScheduler: datamodule.num_pred_steps_val_test not found. Validation steps may not update."
            )

        rank_zero_info("FineTuningScheduler: Initializing...")

        # Store the original limit_train_batches
        self.original_limit_train_batches = trainer.limit_train_batches
        self.original_limit_val_batches = trainer.limit_val_batches

        # Initialize model with current autoregressive step
        pl_module.num_pred_steps_train = self.current_autoregressive_steps
        trainer.datamodule.num_pred_steps_train = self.current_autoregressive_steps

        if hasattr(
            trainer.datamodule, "num_pred_steps_val_test"
        ):  # Check again due to earlier warn
            trainer.datamodule.num_pred_steps_val_test = (
                self.current_autoregressive_steps
            )

        if self.steps_increment > 0:
            num_step_increases = (
                (
                    (
                        self.max_autoregressive_steps
                        - (self.init_autoregressive_steps + self.steps_increment)
                    )
                )
                // self.steps_increment
            ) + 1
            self.finetuning_steps_calculated = (
                num_step_increases * self.update_frequency
            )
        else:
            self.finetuning_steps_calculated = 0

        # Validate training steps configuration
        if hasattr(
            self, "validate_total_steps"
        ) and trainer.estimated_stepping_batches != float("inf"):
            self.validate_total_steps(trainer.estimated_stepping_batches)

        # Reset phase step counter
        self.steps_completed_in_phase = 0
        self._log_current_steps(trainer, pl_module)
        rank_zero_info(
            f"FineTuningScheduler: Initialized. current_autoregressive_steps={self.current_autoregressive_steps}, "
            f"pretraining_steps={self.pretraining_steps}, update_frequency={self.update_frequency}."
        )

    # @override
    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self._reset_train_batch_limit_on_epoch_start:
            trainer.limit_train_batches = self.original_limit_train_batches
            rank_zero_info(
                f"FineTuningScheduler: Reset limit_train_batches to {trainer.limit_train_batches} at train epoch {trainer.current_epoch} start."
            )
            self._reset_train_batch_limit_on_epoch_start = False

        ds_train_steps = trainer.datamodule.num_pred_steps_train
        model_train_steps = pl_module.num_pred_steps_train
        if ds_train_steps != model_train_steps:
            rank_zero_warn(
                f"FineTuningScheduler: Mismatch at train epoch start! "
                f"Dataset num_pred_steps: {ds_train_steps}, Model num_pred_steps_train: {model_train_steps}. "
                f"Model will use {model_train_steps}, Dataloader will use {ds_train_steps}."
            )
        rank_zero_info(
            f"FineTuningScheduler: Train Epoch {trainer.current_epoch}, "
            f"Effective Dataset num_pred_steps: {ds_train_steps}"
        )
        self._log_current_steps(trainer, pl_module)

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self._reset_val_batch_limit_on_epoch_start:
            trainer.limit_val_batches = self.original_limit_val_batches
            rank_zero_info(
                f"FineTuningScheduler: Reset limit_val_batches to {trainer.limit_val_batches} at val epoch {trainer.current_epoch} start."
            )
            self._reset_val_batch_limit_on_epoch_start = False

        if hasattr(trainer.datamodule, "num_pred_steps_val_test"):
            target_val_steps = (
                self.current_autoregressive_steps
            )  # Align with training's current target
            if trainer.datamodule.num_pred_steps_val_test != target_val_steps:
                # old_val_dssteps = trainer.datamodule.num_pred_steps_val_test
                trainer.datamodule.num_pred_steps_val_test = target_val_steps

        self._log_current_steps(trainer, pl_module)

    def on_train_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # trainer.global_step is 1-indexed (updated before this hook)

        # Pretraining Phase
        if trainer.global_step < self.pretraining_steps:
            # This ensures that during pretraining, steps are strictly init_autoregressive_steps
            self._update_train_settings(
                trainer,
                pl_module,
                self.init_autoregressive_steps,
                is_pretraining_phase=True,
            )

            self._log_current_steps(trainer, pl_module)
            return

        # Finetuning Phase (trainer.global_step > self.pretraining_steps)
        if trainer.global_step >= self.pretraining_steps:
            if trainer.global_step == self.pretraining_steps:
                rank_zero_info(
                    f"FineTuningScheduler: Pretraining phase ended at global_step {trainer.global_step}."
                )
                # trainer.relo ad_dataloaders_every_n_epochs = 1

            if self.steps_increment == 0:  # No further increments planned
                self._log_current_steps(trainer, pl_module)
                return

            finetuning_steps_so_far = trainer.global_step - self.pretraining_steps

            if finetuning_steps_so_far >= 0 and (
                finetuning_steps_so_far % self.update_frequency == 0
            ):
                if self.current_autoregressive_steps < self.max_autoregressive_steps:
                    new_potential_steps = (
                        self.current_autoregressive_steps + self.steps_increment
                    )
                    new_steps = min(new_potential_steps, self.max_autoregressive_steps)

                    if new_steps > self.current_autoregressive_steps:
                        self._update_train_settings(trainer, pl_module, new_steps)
                        trainer.datamodule.setup("fit")

        self._log_current_steps(trainer, pl_module)

    def _update_train_settings(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        new_steps: int,
        is_pretraining_phase: bool = False,
    ) -> None:
        old_steps_in_callback = self.current_autoregressive_steps
        old_steps_in_model = pl_module.num_pred_steps_train
        old_steps_in_dataset = trainer.datamodule.num_pred_steps_train
        # Check if an update is actually needed
        if (
            old_steps_in_callback == new_steps
            and old_steps_in_model == new_steps
            and old_steps_in_dataset == new_steps
        ):
            return  # Already consistent

        self.current_autoregressive_steps = new_steps
        pl_module.num_pred_steps_train = new_steps
        trainer.datamodule.num_pred_steps_train = new_steps
        trainer.datamodule.num_pred_steps_val_test = new_steps
        # trainer._data_connector._request_dataloader_reload = True

        rank_zero_info(
            f"FineTuningScheduler: Training autoregressive steps updated. "
            f"Callback: {old_steps_in_callback} -> {self.current_autoregressive_steps}. "
            f"Model: {old_steps_in_model} -> {pl_module.num_pred_steps_train}. "
            f"Dataset Target: {trainer.datamodule.num_pred_steps_train}. "
            f"Global step: {trainer.global_step}."
        )

        if (
            not is_pretraining_phase
            and self.batches_per_step_increase is not None
            and new_steps > old_steps_in_model
        ):
            if (
                trainer.limit_train_batches != self.batches_per_step_increase
            ):  # Avoid redundant sets
                trainer.limit_train_batches = self.batches_per_step_increase
                self._reset_train_batch_limit_on_epoch_start = True
                rank_zero_info(
                    f"FineTuningScheduler: Set limit_train_batches to {self.batches_per_step_increase}. "
                    f"Will reset at next train epoch start."
                )

    def _log_current_steps(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self.logging_interval == "None" or not trainer.loggers:
            return

        # Avoid logging too frequently if logging_interval is "step"
        if self.logging_interval == "step":
            if (
                trainer.global_step == self._last_logged_step
                and trainer.global_step > 0
            ):  # trainer.global_step can be 0 initially
                return

        metrics_to_log = {
            "autoregressive_steps": float(self.current_autoregressive_steps),
        }

        for logger in trainer.loggers:
            logger.log_metrics(metrics_to_log, step=trainer.global_step)

        self._last_logged_step = trainer.global_step

    def validate_total_steps(self, trainer_total_steps: int) -> None:
        """
        Validate that the total training steps in Trainer match pretraining + calculated finetuning steps.
        """
        if self.finetuning_steps_calculated is None:
            rank_zero_warn(
                "FineTuningScheduler: Calculated finetuning_steps not available for validation."
            )
            return

        expected_total_curriculum_steps = (
            self.pretraining_steps + self.finetuning_steps_calculated
        )

        # If batches_per_step_increase is used, the actual number of batches might be different
        # This validation is more about the curriculum length in terms of update_frequency periods
        if trainer_total_steps < expected_total_curriculum_steps:
            rank_zero_warn(
                f"FineTuningScheduler: Trainer's total steps ({trainer_total_steps}) is less than "
                f"the estimated curriculum steps ({expected_total_curriculum_steps} = "
                f"{self.pretraining_steps} pretrain + {self.finetuning_steps_calculated} finetune). "
                f"The curriculum may not complete."
            )
        # If trainer_total_steps is much larger, it's also worth a note.
        elif (
            trainer_total_steps
            > expected_total_curriculum_steps + self.update_frequency
        ):  # Allow some slack
            rank_zero_info(
                f"FineTuningScheduler: Trainer's total steps ({trainer_total_steps}) is greater than "
                f"the estimated curriculum steps ({expected_total_curriculum_steps}). Training will continue after curriculum."
            )

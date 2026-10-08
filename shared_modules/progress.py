"""Training progress with wall-clock case throughput."""

from time import perf_counter

from pytorch_lightning.callbacks import TQDMProgressBar


class CasesPerSecondProgressBar(TQDMProgressBar):
    """Show average training cases/s since the start of the current epoch.

    Counts the actual image batch size, including partial batches, independently
    of optimizer steps or sliding-window batches. Timing includes data loading
    and augmentation, but excludes setup/caching and validation pauses.
    Distributed runs report the displayed rank's local rate as cases/s/GPU.
    No extra device synchronization or distributed communication is performed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._epoch_active = False
        self._started_at = None
        self._paused_at = None
        self._paused_seconds = 0.0
        self._cases = 0
        self._cases_per_second = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_active = True
        self._started_at = perf_counter()
        self._paused_at = None
        self._paused_seconds = 0.0
        self._cases = 0
        self._cases_per_second = None
        super().on_train_epoch_start(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._cases += int(batch["image"].shape[0])
        self._update_rate()
        # Update the rate before the base callback refreshes the postfix.
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_start(self, trainer, pl_module):
        if self._epoch_active and self._paused_at is None:
            self._paused_at = perf_counter()
        super().on_validation_start(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        if self._paused_at is not None:
            self._paused_seconds += perf_counter() - self._paused_at
            self._paused_at = None
        super().on_validation_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        self._update_rate()
        super().on_train_epoch_end(trainer, pl_module)
        self._epoch_active = False

    def _update_rate(self):
        if self._started_at is None:
            return
        end = self._paused_at if self._paused_at is not None else perf_counter()
        elapsed = end - self._started_at - self._paused_seconds
        if elapsed > 0:
            self._cases_per_second = self._cases / elapsed

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        if self._epoch_active and self._cases_per_second is not None:
            key = "cases/s/GPU" if trainer.world_size > 1 else "cases/s"
            metrics[key] = f"{self._cases_per_second:.2f}"
        return metrics

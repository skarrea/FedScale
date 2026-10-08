from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
import torch
from torch.utils.data import DataLoader, Dataset

from shared_modules.progress import CasesPerSecondProgressBar


class ThroughputTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for hook in ("on_train_epoch_start", "on_train_batch_end", "on_train_epoch_end",
                     "on_validation_start", "on_validation_end"):
            self.stack.enter_context(patch.object(TQDMProgressBar, hook))
        self.stack.enter_context(patch.object(TQDMProgressBar, "get_metrics",
                                             side_effect=lambda *args: {"existing_metric": 1}))
        self.clock = self.stack.enter_context(patch("shared_modules.progress.perf_counter"))
        self.bar = CasesPerSecondProgressBar()
        self.trainer = SimpleNamespace(world_size=1)
        self.clock.return_value = 0.0
        self.bar.on_train_epoch_start(self.trainer, None)

    def batch_end(self, count, time):
        self.clock.return_value = time
        self.bar.on_train_batch_end(self.trainer, None, None,
                                    {"image": torch.empty(count, 1)}, 0)

    def test_counts_cases_including_partial_batch(self):
        self.batch_end(4, 2.0)
        self.assertEqual(self.bar.get_metrics(self.trainer, None)["cases/s"], "2.00")
        self.batch_end(1, 4.0)
        self.assertEqual(self.bar._cases, 5)
        self.assertEqual(self.bar.get_metrics(self.trainer, None),
                         {"existing_metric": 1, "cases/s": "1.25"})

    def test_validation_is_excluded_but_between_batch_wait_is_included(self):
        self.batch_end(4, 2.0)
        self.bar.on_validation_start(self.trainer, None)
        self.clock.return_value = 102.0
        self.bar.on_validation_end(self.trainer, None)
        self.batch_end(4, 104.0)
        self.assertEqual(self.bar.get_metrics(self.trainer, None)["cases/s"], "2.00")
        self.batch_end(2, 110.0)
        self.assertEqual(self.bar.get_metrics(self.trainer, None)["cases/s"], "1.00")

    def test_epoch_reset_and_distributed_label(self):
        self.trainer.world_size = 2
        self.batch_end(4, 2.0)
        self.assertEqual(self.bar.get_metrics(self.trainer, None),
                         {"existing_metric": 1, "cases/s/GPU": "2.00"})
        self.bar.on_train_epoch_end(self.trainer, None)
        self.assertEqual(self.bar.get_metrics(self.trainer, None), {"existing_metric": 1})
        self.clock.return_value = 200.0
        self.bar.on_train_epoch_start(self.trainer, None)
        self.assertEqual(self.bar._cases, 0)
        self.assertEqual(self.bar.get_metrics(self.trainer, None), {"existing_metric": 1})
        self.batch_end(1, 201.0)
        self.assertEqual(self.bar.get_metrics(self.trainer, None)["cases/s/GPU"], "1.00")


class CaseDataset(Dataset):
    def __len__(self):
        return 7

    def __getitem__(self, index):
        return {"image": torch.tensor([float(index)])}


class TinyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        return self.linear(batch["image"]).square().mean()

    def validation_step(self, batch, batch_idx):
        return self.linear(batch["image"]).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)


class RecordingProgressBar(CasesPerSecondProgressBar):
    def __init__(self):
        super().__init__()
        self.epoch_case_counts = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_case_counts.append(self._cases)
        super().on_train_epoch_end(trainer, pl_module)


class ProgressIntegrationTests(unittest.TestCase):
    def test_lightning_displays_rate_with_validation_and_gradient_accumulation(self):
        bar = RecordingProgressBar()
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            trainer = pl.Trainer(
                accelerator="cpu", devices=1, max_epochs=2,
                logger=False, enable_checkpointing=False, enable_model_summary=False,
                callbacks=[bar], accumulate_grad_batches=2,
                num_sanity_val_steps=0, val_check_interval=1, limit_val_batches=1,
            )
            trainer.fit(TinyModel(), DataLoader(CaseDataset(), batch_size=3),
                        DataLoader(CaseDataset(), batch_size=1))
        self.assertEqual(bar.epoch_case_counts, [7, 7])
        self.assertEqual(trainer.global_step, 4)
        self.assertIn("cases/s=", output.getvalue())
        self.assertGreater(bar._paused_seconds, 0)
        self.assertGreater(bar._cases_per_second, 0)


if __name__ == "__main__":
    unittest.main()

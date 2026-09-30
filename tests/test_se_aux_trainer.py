"""Synthetic CUDA integration checks for the existing trainer's two base losses."""

import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from easydict import EasyDict
import torch
from torch import nn
from torch.utils.data import DataLoader

import trainer as trainer_module
from networks.vit_seg_modeling import Encoder


class _TinySegmentationModel(nn.Module):
    """Use the actual encoder/SE branches with an inexpensive spatial head."""

    def __init__(self):
        super().__init__()
        config = EasyDict(
            hidden_size=128,
            transformer=dict(num_layers=2, num_heads=4, mlp_dim=256,
                             attention_dropout_rate=0.0, dropout_rate=0.0),
            use_se_block=True, se_aux_loss=True, se_pooling_mode="rms",
            use_shsa=False, use_alternate_shsa=False, use_ats=False,
            use_gumbel_topk=False, topk_attn=0.0, verbose=False,
        )
        self.transformer = nn.Module()
        self.transformer.encoder = Encoder(config, vis=False)
        self.head = nn.Linear(128, 3)
        self.aux_requests = []

    def forward(self, image, return_se_aux=False):
        self.aux_requests.append(return_se_aux)
        batch, _, height, width = image.shape
        result = self.transformer.encoder(image.flatten(2).transpose(1, 2), return_se_aux=return_se_aux)
        encoded, attention, _, gates = result[:4]
        logits = self.head(encoded).transpose(1, 2).reshape(batch, 3, height, width)
        output = (logits, attention, [], gates)
        return output + (result[-1],) if return_se_aux and self.training else output


@unittest.skipUnless(torch.cuda.is_available(), "The existing trainer requires CUDA tensors.")
class TrainerIntegrationTests(unittest.TestCase):
    def test_ce_dice_and_bu_joint_steps(self):
        for use_bu_loss in (False, True):
            with self.subTest(use_bu_loss=use_bu_loss), tempfile.TemporaryDirectory(prefix="se_aux_trainer_") as directory:
                torch.manual_seed(13)
                model = _TinySegmentationModel().cuda()
                se_before = [parameter.detach().clone() for parameter in model.transformer.encoder.SELayer.parameters()]
                main_before = model.head.weight.detach().clone()
                samples = [
                    {"image": torch.randn(128, 4, 4), "label": torch.randint(0, 3, (4, 4)), "case_name": f"synthetic_{i}"}
                    for i in range(2)
                ]
                args = SimpleNamespace(
                    se_aux_loss=True, use_se_block=True, se_aux_weight=0.1,
                    use_kd=False, use_bu_loss=use_bu_loss, base_lr=0.01,
                    num_classes=3, batch_size=2, n_gpu=1, seed=13,
                    dataloader_num_workers=0, dataset="Synthetic", ckpt_filename="smoke",
                    ckpt_dir=directory, tensorboard_run_dir=str(Path(directory) / "events"),
                    lambda_=0.5, verbose=False, create_heatmaps=False,
                    heatmaps_dir=str(Path(directory) / "heatmaps"), max_epochs=1,
                    best_checkpoint_start_epoch=20, learn_tau=False, tau=1.0,
                    tau_min=0.001, buloss_option="B", distance_map_type="unsigned",
                    alpha=1.0, boundary_radius=1,
                )
                scalars = {}
                real_writer = trainer_module.SummaryWriter

                class RecordingWriter(real_writer):
                    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
                        scalars[tag] = float(scalar_value)
                        return super().add_scalar(tag, scalar_value, global_step, *args, **kwargs)

                root_logger = logging.getLogger()
                old_handlers = list(root_logger.handlers)
                old_level = root_logger.level
                try:
                    with patch.object(trainer_module, "_build_datasets", return_value=(samples, samples, "synthetic")), \
                         patch.object(trainer_module, "_make_validation_loader", return_value=DataLoader(samples, batch_size=2)), \
                         patch.object(trainer_module, "_validate", return_value={"loss": 0.5, "loss_ce": 0.4, "loss_dice": 0.6, "mean_dice": 0.25}), \
                         patch.object(trainer_module, "SummaryWriter", RecordingWriter), \
                         patch.object(trainer_module.logging, "info", wraps=logging.info) as console_log:
                        self.assertEqual(trainer_module.trainer(args, model, directory), "Training Finished!")
                        self.assertTrue(any("base_loss: %f, se_aux_loss: %f, se_aux_weighted: %f, total_loss: %f" in str(call.args[0]) for call in console_log.call_args_list))
                finally:
                    for handler in list(root_logger.handlers):
                        if handler not in old_handlers:
                            root_logger.removeHandler(handler)
                            handler.close()
                    root_logger.setLevel(old_level)

                self.assertEqual(model.aux_requests, [True])
                self.assertTrue(any(not torch.equal(before, after) for before, after in zip(se_before, model.transformer.encoder.SELayer.parameters())))
                self.assertFalse(torch.equal(main_before, model.head.weight))
                for prefix in ("info", "epoch"):
                    self.assertAlmostEqual(scalars[f"{prefix}/se_aux_weighted"], 0.1 * scalars[f"{prefix}/se_aux_loss"], places=7)
                    self.assertAlmostEqual(scalars[f"{prefix}/total_loss"], scalars[f"{prefix}/base_loss"] + scalars[f"{prefix}/se_aux_weighted"], places=6)
                self.assertTrue(list(Path(args.tensorboard_run_dir).glob("events.out.tfevents.*")))
                self.assertEqual(len(list(Path(directory).glob("*.pth"))), 1)
                self.assertEqual(len(list(Path(directory).glob("*.pth.se_aux.json"))), 1)


if __name__ == "__main__":
    unittest.main()

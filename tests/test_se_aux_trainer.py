"""Synthetic checks for the existing trainer and Gumbel/BU/SE composition."""

import copy
import json
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
from src.se_aux_training import compute_se_aux_loss


class _TinySegmentationModel(nn.Module):
    """Use the actual encoder/SE branches with an inexpensive spatial head."""

    def __init__(self, use_gumbel_topk=False, sampling_mode="manual"):
        super().__init__()
        config = EasyDict(
            hidden_size=128,
            transformer=dict(num_layers=2, num_heads=4, mlp_dim=256,
                             attention_dropout_rate=0.0, dropout_rate=0.0),
            use_se_block=True, se_aux_loss=True, se_pooling_mode="rms",
            use_shsa=False, use_alternate_shsa=False, use_ats=False,
            use_gumbel_topk=use_gumbel_topk,
            topk_attn=0.5 if use_gumbel_topk else 0.0,
            gumbel_sampling_mode=sampling_mode, verbose=False,
        )
        self.transformer = nn.Module()
        self.transformer.encoder = Encoder(config, vis=False)
        self.head = nn.Linear(128, 3)
        self.aux_requests = []

    def forward(self, image, return_se_aux=False):
        self.aux_requests.append(return_se_aux)
        batch, _, height, width = image.shape
        result = self.transformer.encoder(image.flatten(2).transpose(1, 2), return_se_aux=return_se_aux)
        encoded, attention, kept_indices, gates = result[:4]
        if kept_indices is not None:
            full = encoded.new_zeros(batch, height * width, encoded.shape[-1])
            encoded = full.scatter(1, kept_indices.unsqueeze(-1).expand_as(encoded), encoded)
        logits = self.head(encoded).transpose(1, 2).reshape(batch, 3, height, width)
        output = (logits, attention, [], gates)
        return output + (result[-1],) if return_se_aux and self.training else output


@unittest.skipUnless(torch.cuda.is_available(), "The existing trainer requires CUDA tensors.")
class TrainerIntegrationTests(unittest.TestCase):
    def test_ce_dice_and_bu_joint_steps(self):
        for use_bu_loss, use_gumbel_topk, learn_tau in (
            (False, False, False), (True, False, False),
            (True, True, False), (True, True, True),
        ):
            with self.subTest(bu=use_bu_loss, gumbel=use_gumbel_topk, learn_tau=learn_tau), tempfile.TemporaryDirectory(prefix="se_aux_trainer_") as directory:
                torch.manual_seed(13)
                model = _TinySegmentationModel(use_gumbel_topk=use_gumbel_topk).cuda()
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
                    best_checkpoint_start_epoch=20, learn_tau=learn_tau,
                    tau=None if learn_tau else 1.0, tau_init=1.0,
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
                metadata_files = list(Path(directory).glob("*.pth.se_aux.json"))
                self.assertEqual(len(metadata_files), 1)
                metadata = json.loads(metadata_files[0].read_text())
                self.assertEqual(metadata["pooling_mode"], "rms")
                self.assertEqual(metadata["num_blocks"], 2)
                self.assertEqual(metadata["target_config"], model.transformer.encoder.se_aux_config)
                if use_bu_loss:
                    self.assertGreater(scalars["info/tau"], args.tau_min)


class GumbelBUCompositionTests(unittest.TestCase):
    def test_detached_auxiliary_preserves_bu_gradients_with_fixed_and_learned_tau(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        for sampling_mode in ("dist", "manual"):
            for learn_tau in (False, True):
                with self.subTest(sampling_mode=sampling_mode, learn_tau=learn_tau):
                    torch.manual_seed(13)
                    model = _TinySegmentationModel(True, sampling_mode).train()
                    reference = copy.deepcopy(model)
                    args = SimpleNamespace(
                        buloss_option="B", distance_map_type="unsigned", learn_tau=learn_tau,
                        tau=None if learn_tau else 1.0, tau_init=1.0, tau_min=0.001,
                        alpha=1.0, boundary_radius=1, lambda_=0.5, num_classes=3,
                    )
                    bu_loss = trainer_module.BULoss(args=args)
                    reference_bu = copy.deepcopy(bu_loss)
                    images = torch.randn(2, 128, 8, 8)
                    labels = torch.zeros(2, 8, 8, dtype=torch.long)
                    labels[:, 4:, :4] = 1
                    labels[:, 4:, 4:] = 2
                    rng_state = torch.get_rng_state()
                    logits, attention, _, gates, targets = model(images, return_se_aux=True)
                    self.assertEqual([a.shape[-2] for a in attention], [32, 16])
                    self.assertTrue(all(not t.requires_grad and t.grad_fn is None for t in targets))
                    aux = compute_se_aux_loss(gates, targets, expected_blocks=2)
                    aux.backward(retain_graph=True)
                    se_parameters = list(model.transformer.encoder.SELayer.parameters())
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in se_parameters), 0)
                    self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if ".SELayer." not in name))
                    model.zero_grad(set_to_none=True)
                    (bu_loss(logits, labels) + 0.1 * aux).backward()
                    torch.set_rng_state(rng_state)
                    reference_logits = reference(images)[0]
                    torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)
                    reference_bu(reference_logits, labels).backward()
                    for (name, parameter), reference_parameter in zip(model.named_parameters(), reference.parameters()):
                        if ".SELayer." in name:
                            self.assertIsNone(reference_parameter.grad)
                        elif parameter.grad is not None or reference_parameter.grad is not None:
                            torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)
                    if learn_tau:
                        self.assertTrue(torch.isfinite(bu_loss.rho.grad))
                        self.assertGreater(bu_loss.rho.grad.abs().item(), 0)
                        torch.testing.assert_close(bu_loss.rho.grad, reference_bu.rho.grad, rtol=0, atol=0)
                    else:
                        self.assertEqual(bu_loss.get_tau().item(), 1.0)


if __name__ == "__main__":
    unittest.main()

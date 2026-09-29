The diagnostic is implemented in [`tools/check_se_gradient_flow.py`](../tools/check_se_gradient_flow.py). No existing model, trainer, quantizer, or experiment files were modified.

The suspected disconnection does **not** describe ordinary SE in the current code. Ordinary SE feeds its scaled features into the segmentation stream. The existing `se_calib_only` mode computes gates from detached features and discards the scaled features; with only segmentation loss, the measurements support disconnection in that mode.

All measurements below used commit `ad3adb3b1112edf6c120bd769c0be064e3037c70`, the actual 106,161,817-parameter R50–ViT-B/16 model, 12 Transformer layers, 12 heads, hidden width 768, SE enabled, and token reduction disabled. Inputs were synthetic FP32 images `[1,3,224,224]` and integer masks `[1,224,224]` with 9 classes; seed 1234; three training steps; no checkpoint or pretrained downloads. Loss was `0.5 * CrossEntropyLoss + 0.5 * utils.DiceLoss(softmax=True)`. SGD included all model parameters with initial LR 0.01, momentum 0.9, and weight decay 0.0001. Its post-step polynomial schedule used three steps as the diagnostic horizon. No auxiliary loss was used.

| Experiment | SE gradients each step | Gate gradients each step | SE tensors updated each step | Largest SE update across steps |
|---|---:|---:|---:|---:|
| Ordinary SE, CUDA | 24/24 nonzero | 12/12 nonzero | 24/24 | 2.08616e-6 |
| Calibration-only SE, CUDA | 24/24 `None` | 12/12 `None` | 0/24 | 0 |

Four ordinary Transformer parameters had nonzero gradients and changed on every step in both experiments. The script discovers SE parameters through actual `SELayer` module ownership; the counts above are measured, not required constants. Per-parameter names, shapes, `requires_grad`, gradient states/norms, and maximum updates are in the detailed JSON and verbose logs.

Evaluation used the same trained model and same inputs/labels within each experiment, restoring all parameters, buffers, bypass flags, module training modes, and RNG state. Comparisons use `abs(candidate-reference) <= 1e-6 + 1e-5 * abs(reference)` elementwise.

| CUDA comparison | Maximum absolute logit difference | Absolute total-loss difference | Logits within tolerance | Loss within tolerance |
|---|---:|---:|---|---|
| Ordinary SE: unchanged repeat | 0 | 0 | yes | yes |
| Ordinary SE: SE parameters perturbed | 7.65256584e-4 | 8.34465027e-7 | no | yes |
| Ordinary SE: SE bypassed | 7.65696168e-4 | 4.76837158e-7 | no | yes |
| Ordinary SE: restored repeat | 0 | 0 | yes | yes |
| Calibration-only: unchanged repeat | 0 | 0 | yes | yes |
| Calibration-only: SE parameters perturbed | 0 | 0 | yes | yes |
| Calibration-only: SE bypassed | 0 | 0 | yes | yes |
| Calibration-only: restored repeat | 0 | 0 | yes | yes |

Perturbation changed all 12 returned gates beyond tolerance in both modes: maximum gate changes were 0.60659230 (ordinary) and 0.70561236 (calibration-only). Bypass returned no gates. Every experimental control passed, including exact model-state restoration. Ordinary SE's segmentation-loss connection is directly demonstrated by nonzero gradients; its small scalar loss differences alone remain within the documented comparison tolerance.

A full 224px, three-step CPU run of ordinary SE also completed, invoked from `/tmp` to check independence from the working directory. All controls passed. Its perturbation/bypass maximum logit differences were 7.53566623e-4 / 6.87554479e-4; loss differences were 1.19209290e-7 / 3.57627869e-7. Unchanged/restored repeats were exactly equal. CPU and CUDA runs are independent experiments; the ablation comparisons always use a single model within each run.

Executed artifacts:

- [Ordinary CUDA JSON](se_gradient_flow_cuda.json) and [verbose log](se_gradient_flow_cuda.log).
- [Calibration-only CUDA JSON](se_gradient_flow_calibration_cuda.json) and [verbose log](se_gradient_flow_calibration_cuda.log).
- [Ordinary CPU JSON](se_gradient_flow_cpu.json) and [log](se_gradient_flow_cpu.log).
- [Checkpoint validation](se_checkpoint_validation.json): actual full-model raw and wrapped `module.`-prefixed state dictionaries loaded strictly; missing keys, unexpected keys, and shape mismatches were each rejected, with complete model-state hashes unchanged on rejection.

Syntax/CLI checks passed. Additional interpretation checks confirmed failed Transformer controls, nonfinite evaluation gates, or ineffective gate perturbation produce an inconclusive result. A Python environment missing torch returned exit code 1 and explicit error JSON. No historical trained checkpoint, quantization run, real-data training, or calibration-only CPU run was performed. The checkpoint tests used freshly generated temporary fixtures, not historical checkpoints.

The working runtime was Python 3.12 / PyTorch 2.5.1 on NVIDIA RTX A6000 GPUs. Missing repository dependencies were installed in `/tmp/transunet-se-diagnostic-venv`, inheriting the existing PyTorch installation without changing it. This PyTorch CUDA CE kernel lacks a strictly deterministic implementation. The script requests deterministic kernels with warning mode, disables TF32, controls RNG state, and checks repeatability experimentally; both evaluation repeats were exactly equal in every executed experiment. JSON/log artifacts exist locally but are ignored by the repository's existing `*.json` / `*.log` rules.

Relevant source locations:

- [`networks/se_block.py:42`](../networks/se_block.py#L42): scales features by the returned gate.
- [`networks/vit_seg_modeling.py:432`](../networks/vit_seg_modeling.py#L432): calibration-only mode detaches input and discards scaled features.
- [`networks/vit_seg_modeling.py:434`](../networks/vit_seg_modeling.py#L434): ordinary SE updates `hidden_states`, which proceeds through encoder normalization, decoder, and segmentation head.
- [`networks/vit_seg_modeling.py:385`](../networks/vit_seg_modeling.py#L385): supported `drop_se_block` bypass.
- [`trainer.py:209`](../trainer.py#L209), [`trainer.py:245`](../trainer.py#L245), [`utils.py:53`](../utils.py#L53): actual logits and CE + Dice computation. Optional SE auxiliary loss is separately handled at trainer.py:247–254 and excluded here.
- [`src/quantize.py:925`](../src/quantize.py#L925), [`src/quantize.py:1049`](../src/quantize.py#L1049): subsequent calibration can consume gates and use them in activation saliency.

`grad=None` is different from a computed zero gradient. `requires_grad=True` or a `grad_fn` establishes autograd tracking, not necessarily a path to segmentation loss. Unchanged SE parameters do not imply constant gates because backbone features can change. A segmentation-disconnected branch may still affect later quantization. Synthetic diagnostics establish the tested computation graph's behavior, not historical training or quantization results. State dictionaries also do not establish historical runtime flags; select the intended mode explicitly.

From the repository root, with the repository dependencies installed in the active Python environment:

```bash
# No checkpoint; auto-select CUDA when available, otherwise CPU.
python tools/check_se_gradient_flow.py --device auto --steps 3 \
  --json-output diagnostics/se_gradient_flow.json

# Strictly load a trained checkpoint; match its class count and input resolution.
python tools/check_se_gradient_flow.py --device auto --steps 3 \
  --checkpoint /path/to/checkpoint.pth --num-classes 9 --img-size 224 \
  --json-output diagnostics/se_gradient_flow_checkpoint.json

# Verbose per-parameter and gate logging to both console and a file.
python tools/check_se_gradient_flow.py --device auto --steps 3 --verbose \
  --json-output diagnostics/se_gradient_flow_verbose.json \
  --log-file diagnostics/se_gradient_flow_verbose.log

# Exercise the existing calibration-only mode with segmentation loss alone.
python tools/check_se_gradient_flow.py --device auto --steps 3 --se-calib-only \
  --verbose --json-output diagnostics/se_gradient_flow_calibration.json \
  --log-file diagnostics/se_gradient_flow_calibration.log
```

Use `--device cpu` for CPU execution. Add `--se-calib-only` to the checkpoint command if that is the intended runtime mode. The environment used here can be selected by replacing `python` with `/tmp/transunet-se-diagnostic-venv/bin/python` while that temporary environment remains available. The script returns 0 for a completed experiment with valid controls, 2 for inconclusive controls, and 1 for an execution or checkpoint error.

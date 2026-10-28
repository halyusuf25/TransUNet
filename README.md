# TransUNet
This repo holds code for [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf)

## 📰 News
- [7/26/2024] TransUNet, which supports both 2D and 3D data and incorporates a Transformer encoder and decoder, has been featured in the journal Medical Image Analysis ([link](https://www.sciencedirect.com/science/article/pii/S1361841524002056)).
```bibtex
@article{chen2024transunet,
  title={TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers},
  author={Chen, Jieneng and Mei, Jieru and Li, Xianhang and Lu, Yongyi and Yu, Qihang and Wei, Qingyue and Luo, Xiangde and Xie, Yutong and Adeli, Ehsan and Wang, Yan and others},
  journal={Medical Image Analysis},
  pages={103280},
  year={2024},
  publisher={Elsevier}
}
```

- [10/15/2023] 🔥 3D version of TransUNet is out! Our 3D TransUNet surpasses nn-UNet with 88.11% Dice score on the BTCV dataset and outperforms the top-1 solution in the BraTs 2021 challenge and secure the second place in BraTs 2023 challenge. Please take a look at the [code](https://github.com/Beckschen/3D-TransUNet/tree/main) and [paper](https://arxiv.org/abs/2310.07781).


## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data (All data are available!)

All data are available so no need to send emails for data. Please use the [BTCV preprocessed data](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing) and [ACDC data](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4?usp=drive_link).

### 3. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on synapse dataset. The batch size can be reduced to 12 or 6 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

### Auxiliary SE training

Enable `--use_se_block --se_aux_loss` together to train the existing per-block SE
predictors alongside segmentation in one optimizer step. `--se_aux_loss` alone
is rejected. Without either flag behavior is unchanged; `--use_se_block` alone
keeps the legacy mean pooling and loss. Auxiliary SE training is not supported
with KD. The default `--se_aux_weight 0.1` is an initial tunable value, not an
experimentally established optimum; it must be finite and positive when enabled.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16 \
  --use_se_block --se_aux_loss --se_aux_weight 0.1 --ckpt se_aux
```

The predictor pools detached post-block features with float32 RMS,
`sqrt(mean(h.float() ** 2, dim=1))`, then uses the unchanged SE MLP to produce
`[B,1,C]` gates. The segmentation feature stream is unchanged. Auxiliary gradients
reach only SE parameters; segmentation gradients reach only the main model.
The optimizer still uses `model.parameters()` exactly once. Console and
TensorBoard report base loss, raw auxiliary loss, weighted auxiliary loss, and
total loss.

For each projection input `X[B,N,C]` and current linear weight `W[O,C]`, define
`e = sqrt(mean_N(X**2) * mean_O((W - QDQ(W))**2))` and
`target = (e + 1e-12) / (e + mean_C(e) + 2e-12)`. Targets are detached float32
`[B,1,C]` tensors; zero-error inputs give 0.5. Weight QDQ matches AWQ's asymmetric
rounding, clipping, zero points, and constant-group range clamp using **4 bits,
128 input channels per group, and zero_point=True**. Training uses a small
PyTorch-only helper, without importing AWQ or running calibration searches.
Targets are recomputed from current weights each iteration.

Each block target averages the QKV and `ffn.fc1` targets with equal weight. QKV
uses its actual input after `attention_norm` and concatenated query/key/value
weights, or the verified fused QKV weights for Top-K, Gumbel Top-K, and ATS.
FC1 uses its actual input after `ffn_norm`; its token count can differ from QKV.
Partial-channel SHSA has no defined full-channel mapping and is rejected for
auxiliary training. The loss averages exact-shape gate/target MSE over blocks
and over all images/channels, including unequal DataParallel replica batches.
This target is a channelwise quantization-error proxy: the predictor sees RMS
statistics at the block output while its target comes from two earlier inputs.
RMS pooling does not remove this remaining information mismatch.

Checkpoints retain the existing `.pth` state-dict format. Keep each associated
`.pth.se_aux.json` metadata file with its checkpoint: it records pooling and
target settings. `test.py` and the checkpoint benchmark loader restore RMS predictors from this metadata while
disabling auxiliary losses and targets; checkpoints without metadata use legacy
mean pooling. Evaluation also skips targets while a training-configured model
is in `eval()` mode. Calibration verifies that its weight quantization settings
match the recorded target settings and removes SE after quantization.

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16 --use_se_block \
  --quantize --ckpt_dir ckpt --ckpt YOUR_CHECKPOINT.pth
```

The existing AWQ deployment quantizer supports separate query/key/value
attention projections. Its calibration paths and replacement list do not
support the fused Top-K/Gumbel/ATS projections or partial-channel SHSA; training
target support for fused attention does not add deployment support for those
variants. Its calibration input reader also does not currently support
EndoVis2018. Real quantization still requires the existing AWQ/CUDA dependencies.

### Optional Intel Neural Compressor AWQ

`--quantize_backend` defaults to `legacy`; selecting `inc_awq` only takes effect
with `--quantize`. INC uses standard activation-aware AWQ with automatic scaling
and clipping, independent of SE gates. `--use_se_block` still loads SE checkpoint
weights and pooling metadata; the existing inference path removes the predictors
after quantization. The SE target normalization (`eps=1e-12`) is unchanged and is
not passed to INC.

Target version: **neural-compressor 3.10**, which requires **Python >=3.11**.
Install in an environment containing the repository's PyTorch/model dependencies:

```bash
python -m pip install neural-compressor==3.10 prettytable psutil py-cpuinfo pydantic packaging
```

The optional dependency is imported only when this backend runs. No legacy
`awq` package is required. The explicit support packages avoid the unrelated
`torchao` dependency pulled in by INC's `[pt]` extra; no repository dependency
versions are changed. On Python 3.12, the existing `thop` package also needs
`setuptools` for its `distutils` import.

The backend selects only `transformer.encoder.layer.*.attn.query/key/value/out`
for standard attention, or `attn.qkv/proj` for Top-K/Gumbel/ATS, plus `ffn.fc1/fc2`.
It reports every selected name and rejects SHSA, Swin, nonhybrid embeddings,
unknown/replaced encoder layouts, and input widths not divisible by 128 before quantization. The tested
model structure is the repository's ResNet/ViT hybrid with the original encoder,
decoder and segmentation head. Pruning and the encoder's token-index composition
remain active. A temporary adapter exposes the original blocks to INC, so its
first-`ModuleList` discovery cannot select the backbone, SE predictors or decoder.

Settings are asymmetric W4, `group_size=128`, `group_dim=1`, `dtype="int"`,
`use_auto_scale=True`, `use_auto_clip=True`, and `folding=False`. The final
**QDQ representation retains floating-point weights and computation**, including
compensating input scales/`MulLinear` wrappers. INC 3.10's final conversion packs
weights despite AWQ requesting `return_int=False`; the integration materializes
INC's own `recover()` output in floating-point linears inside those wrappers.
This preserves INC's quantized values without a second quantization algorithm.
The resulting model is neither packed INT4 storage
nor accelerated INT4 inference. Benchmarks use the existing floating-point/QDQ
counting path; their `quantized_model` note refers specifically to legacy
`WQLinear` counting. Saved arguments record `quantize_backend=inc_awq`.

Calibration reuses the test split and order-3 image resizing, including optional
frame normalization. `--quantize_calibrate_batch_size` counts **loader batches**
(one whole volume with all slices, or one frame), not images or slices. Every
requested batch is used under `eval()`/`no_grad()`; a short loader raises an error.
For example, with the existing CUDA test pipeline and an SE checkpoint:

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16 \
  --volume_path /path/to/test_vol_h5 --list_dir lists/lists_Synapse \
  --ckpt_dir ckpt --ckpt YOUR_CHECKPOINT.pth --use_se_block \
  --quantize --quantize_backend inc_awq --quantize_calibrate_batch_size 8
```

Omit `--use_se_block` for a baseline checkpoint. The backend itself also supports
CPU calibration/forward; `test.py` retains the repository's CUDA model-loading
and volumetric-inference requirements.

```bash
python -m unittest discover -s tests -p test_inc_awq.py -v
```

These checks cover default arguments, lazy imports without either quantizer
package, explicit layer selection/rejection, and real INC calibration followed
by finite segmentation outputs and pruning-index checks. The real INC check
reports a skip when the optional package is absent.

Initial QDQ-only validation used Python 3.12.14, `neural-compressor==3.10`, PyTorch 2.3.0+cpu,
torchvision 0.18.0+cpu and NumPy 1.26.4 in an isolated environment. Real INC
calibration and segmentation forwards passed for standard, Top-K, Gumbel and
ATS attention, including exact CPU agreement between INC's floating forward
and the materialized QDQ projections. Those initial checks did not include CUDA,
full-dataset accuracy or throughput. See the [INC documentation](https://intel.github.io/neural-compressor/latest/docs/source/Welcome.html)
for the public PyTorch quantization API.

`--quantize --quantize_backend inc_awq_packed` reuses the same INC calibration
and repacks INC's existing integer codes/scales/zero points into the installed
`awq.quantize.qmodule.WQLinear` layout. It preserves compensation wrappers and
uses contiguous FP16 inputs to the INT4-weight CUDA kernels. This path requires
INC 3.10, the existing llm-awq 0.1.0 packing API and a compatible compiled CUDA
extension on SM80+ GPUs, with input widths divisible by 256 and output widths
by 128. It validates dependencies before calibration and never falls back to
QDQ. Both `legacy` and `inc_awq_packed` use the existing WQLinear benchmark
handling; `inc_awq` remains floating QDQ. See
[packed backend validation](docs/inc_awq_packed_validation.md) for dependency
identities, numerical checks, benchmark commands/results and limitations, and
[quantization commands](tools/commands/quantization.txt) for an ACDC example.

Run the focused synthetic checks (no datasets or pretrained weights needed):

```bash
python -m unittest discover -s tests -p 'test_se_aux*.py' -v
```

The suite checks gradients, exact image/channel pairing, current-weight targets,
checkpoint pooling, evaluation and SE removal, attention input mappings, and
actual CE/Dice and BU trainer steps. CUDA, multi-GPU, and installed AWQ reference
checks report explicit skips when their required environment is unavailable.

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Citations


```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L., and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```

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

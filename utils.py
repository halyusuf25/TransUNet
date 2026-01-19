import logging
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import time
from typing import Tuple, Dict, Any
from thop import profile
from fvcore.nn import FlopCountAnalysis

def _make_json_safe(value: Any) -> Any:
    """Recursively convert objects into JSON-serializable types."""
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_make_json_safe(v) for v in value.tolist()]
    if torch.is_tensor(value):
        return _make_json_safe(value.detach().cpu().tolist())
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    return value

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def flatten(input, target, ignore_index):
    num_class = input.size(1)
    input = input.permute(0, 2, 3, 1).contiguous()
    
    input_flatten = input.view(-1, num_class)
    target_flatten = target.view(-1)
    
    mask = (target_flatten != ignore_index)
    input_flatten = input_flatten[mask]
    target_flatten = target_flatten[mask]
    
    return input_flatten, target_flatten


class JaccardLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super(JaccardLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
      
    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        input = torch.nn.functional.softmax(input, dim=1)
        num_classes = input.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (target == c).float()
            input_c = input[:, c]
            
            intersection = (input_c * target_c).sum()
            total = (input_c + target_c).sum()
            union = total - intersection
            IoU = (intersection + self.smooth)/(union + self.smooth)
            
            losses.append(1-IoU)
        
        losses = torch.stack(losses)
        loss = losses.mean()
        return loss


# --- utils.py ---
def calculate_metric_percase(pred, gt, voxelspacing=None):
    """
    General metric helper matching the original Synapse evaluation protocol,
    which treats mutually absent classes as a perfect overlap.
    """
    # both binary arrays (0/1) on input
    P, G = pred.sum() > 0, gt.sum() > 0
    
    if P and G:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxelspacing)  # <<< pass spacing
        iou = metric.binary.jc(pred, gt)
        return float(dice), float(hd95), float(iou)

    if P and not G:
        # predicted spurious organ
        return 0.0, np.nan, 0.0  # or np.nan; but DO NOT make it (1,0)

    if not P and G:
        # missed organ completely
        return 0.0, np.nan, 0.0  # or np.nan

    # both empty: agree on absence
    return 1.0, 0.0, 1.0  # many protocols treat this as perfect agreement for overlap


def calculate_metric_percase_cataract(pred, gt, voxelspacing=None):
    """
    Metric helper that avoids rewarding absent classes with perfect scores.
    Returns NaNs when both prediction and ground-truth are empty so downstream
    aggregation can ignore them via nan-aware reductions.
    """
    P, G = pred.sum() > 0, gt.sum() > 0

    if P and G:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxelspacing)
        iou = metric.binary.jc(pred, gt)
        return float(dice), float(hd95), float(iou)
    # elif P > 0 and G==0:
    #     return 1, 0, 1
    # else:
    #     return 0, 0, 0

    if P and not G:
        return 0.0, np.nan, 0.0

    if not P and G:
        return 0.0, np.nan, 0.0

    return np.nan, np.nan, np.nan
    
def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, dataset='Synapse'):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    net.eval()
    # if len(image.shape) == 3:
    if dataset == 'Synapse':
        prediction = np.zeros_like(label)
        print(f"Processing case {case}, image shape: {image.shape}")
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]    
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            with torch.no_grad():
                outputs, _ , _ ,_= net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    elif dataset == 'Cataract1k':
        # Resize each channel to patch_size and stack
        resized_channels = []
        H, W, C = image.shape
        for ch in range(C):
            slice = image[:, :, ch]
            # Resize to patch_size
            resized_slice = zoom(slice, (patch_size[0]/H, patch_size[1]/W), order=3)
            resized_channels.append(resized_slice)
        # Convert to tensor (1, 3, H, W)
        input = torch.from_numpy(np.stack(resized_channels, axis=0)).unsqueeze(0).float().cuda()
        # Forward pass
        with torch.no_grad():
            outputs, _ , _ = net(input)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()
            # Resize prediction back to original dimensions
            pred = zoom(out, (H/patch_size[0], W/patch_size[1]), order=0)
            prediction = pred  # Direct assignment to 2D array
    else:
        raise ValueError("Unknown dataset")
    
    metric_list = []
    metric_fn = calculate_metric_percase_cataract if dataset == 'Cataract1k' else calculate_metric_percase
    for i in range(1, classes):
        metric_list.append(metric_fn(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list



def evaluate_model_perf(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    throughput_batch_size: int = 64,
    warmup: int = 10,
    iterations: int = 500,
    device: str = None
) -> Dict[str, float]:
    """
    Evaluate model metrics including Parameters, FLOPs, Latency, and Throughput
    
    Args:
        model: PyTorch model to evaluate
        input_size: Input tensor dimensions (channels, height, width)
        throughput_batch_size: Batch size for throughput measurement
        warmup: Number of warmup runs before timing
        iterations: Number of iterations for timing measurements
        device: 'cuda' or 'cpu' (auto-detected if None)
    
    Returns:
        Dictionary with all metrics
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    
    # Generate dummy inputs
    dummy_input = torch.randn((1, *input_size)).to(device)
    throughput_input = torch.randn((throughput_batch_size, *input_size)).to(device)

    # --------------------------
    # 1. Compute Parameters
    # --------------------------
    total_params = sum(p.numel() for p in model.parameters())
    
    # --------------------------
    # 2. Compute FLOPs (for single input)
    # --------------------------
    flops = FlopCountAnalysis(model, dummy_input)
    total_flops = flops.total()

    # --------------------------
    # 3. Measure Latency (batch=1)
    # --------------------------
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_input)

    # Timing
    total_time = 0.0
    for _ in range(iterations):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        if device == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - start
    
    latency_ms = (total_time / iterations) * 1000

    # --------------------------
    # 4. Measure Throughput
    # --------------------------
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(throughput_input)
    
    # Timing
    total_time = 0.0
    for _ in range(iterations):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model(throughput_input)
        if device == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - start
    
    throughput = (throughput_batch_size * iterations) / total_time

    #---------------------------
    # 5. MACs (Multiply-Accumulate Operations)
    # --------------------------
    try:
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        macs = macs / 1e9  # Convert to billions
    except Exception as e:
        print(f"Warning: Could not compute MACs due to {e}. Setting MACs to 0.")
        macs = 0.0

    return {
        "parameters(M)": total_params / 1e6,
        "flops(G)": total_flops / 1e9,
        "latency(ms)": latency_ms,
        "throughput(images/s)": throughput,
        "macs(G)": macs
    }


# compute model size in MB/MiB
def model_size_mb_benchmark(
    model: nn.Module,
    *,
    include_buffers: bool = True,
    use_state_dict: bool = False,
    binary_mebibytes: bool = True,   # True: MiB (1024^2). False: MB (10^6).
    deduplicate_shared_tensors: bool = True,
) -> Dict[str, Any]:
    """
    Estimate the *model size / weight footprint* used in benchmarking,
    following the convention commonly adopted in quantization and
    model-compression research papers.

    --------------------------------------------------------------------
    WHAT THIS FUNCTION MEASURES
    --------------------------------------------------------------------
    This function estimates how many BYTES are required to store the
    model's weights (and optionally buffers) by summing:

        bytes = num_elements × element_size_in_bytes

    across all counted tensors.

    This corresponds to the commonly reported metric:
        "Model size (MB)"  or  "Weight footprint (MB)"

    It reflects changes in numeric precision (e.g., FP32 → INT8 / INT4),
    while keeping the architecture fixed.

    --------------------------------------------------------------------
    OUTPUT DICTIONARY (DETAILED DESCRIPTION)
    --------------------------------------------------------------------
    The function returns a dictionary with the following keys:

    1) total_bytes : int
       ------------------------------------------------------------
       The TOTAL number of bytes required to store all counted tensors.

       Computation:
           sum(t.numel() × t.element_size())  over all counted tensors

       This is the most fundamental quantity; all other size metrics
       are derived from this value.

    2) total_mb : float
       ------------------------------------------------------------
       The same total size expressed in MB or MiB.

       If binary_mebibytes=True (default):
           total_bytes / (1024^2)   → MiB

       If binary_mebibytes=False:
           total_bytes / (10^6)     → MB

       This is the value typically reported in benchmarking tables as:
           "Model size (MB)" or "Model size (MiB)"

    3) unit : str
       ------------------------------------------------------------
       The unit used for total_mb and dtype_mb.

       Possible values:
           - "MiB"  (binary, 1024^2 bytes)
           - "MB"   (decimal, 10^6 bytes)

       Important for paper clarity: always specify which unit you use.

    4) num_tensors_counted : int
       ------------------------------------------------------------
       The number of tensors that were *iterated over* during counting.

       NOTE:
       - This is counted BEFORE deduplication.
       - If the model contains shared/tied weights and
         deduplicate_shared_tensors=True, fewer unique tensors may
         actually contribute to total_bytes.

       This field is mostly diagnostic/debugging information.

    5) dtype_bytes : Dict[str, int]
       ------------------------------------------------------------
       A breakdown of storage size grouped by tensor dtype.

       Example:
           {
               "torch.float32": 84_934_656,
               "torch.int8":     12_345_678
           }

       This is extremely useful for quantization analysis, as it
       explicitly shows how much of the model footprint is coming
       from low-precision vs high-precision tensors.

    6) dtype_mb : Dict[str, float]
       ------------------------------------------------------------
       Same as dtype_bytes, but converted to MB or MiB using the same
       unit as total_mb.

       This is often useful for directly reporting per-dtype memory
       contributions in ablation tables or appendices.

    7) counted_via : str
       ------------------------------------------------------------
       Indicates WHICH tensors were used for counting.

       Possible values:
           - "state_dict"
               Counted tensors from model.state_dict(), which includes:
                 • parameters
                 • persistent buffers (e.g., BatchNorm running stats)

               This is the recommended default for benchmarking,
               especially for quantized models.

           - "parameters(+buffers)"
               Counted model.parameters() and model.buffers().

           - "parameters_only"
               Counted only trainable parameters.

       In most research papers, "state_dict" best matches the notion of
       "model size" as what must be stored or shipped.
     --------------------------------------------------------------------"""

    # Collect tensors
    tensors = []
    if use_state_dict:
        # state_dict contains params + buffers needed for inference
        sd = model.state_dict()
        for _, v in sd.items():
            if torch.is_tensor(v):
                tensors.append(v)
    else:
        tensors.extend(list(model.parameters()))
        if include_buffers:
            tensors.extend(list(model.buffers()))

    # Sum bytes
    total_bytes = 0
    dtype_bytes: Dict[str, int] = {}

    seen = set()
    for t in tensors:
        # bytes for THIS tensor's values
        b = int(t.numel()) * int(t.element_size())

        if deduplicate_shared_tensors:
            # Best-effort: if exact tensor storage is shared, avoid double counting.
            # (Works well for tied weights; rare false negatives for complex views.)
            key = (int(t.data_ptr()), int(t.numel()), int(t.element_size()))
            if key in seen:
                continue
            seen.add(key)

        total_bytes += b
        k = str(t.dtype)
        dtype_bytes[k] = dtype_bytes.get(k, 0) + b

    denom = (1024 ** 2) if binary_mebibytes else (10 ** 6)
    total_mb = total_bytes / denom

    return {
        "total_bytes": total_bytes,
        "total_mb": total_mb,
        "unit": "MiB" if binary_mebibytes else "MB",
        "num_tensors_counted": len(tensors),
        "dtype_bytes": dtype_bytes,
        "dtype_mb": {k: v / denom for k, v in dtype_bytes.items()},
        "counted_via": "state_dict" if use_state_dict else ("parameters(+buffers)" if include_buffers else "parameters_only"),
    }

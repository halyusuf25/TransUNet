import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import time
from typing import Tuple, Dict
from thop import profile

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


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, dataset='Synapse'):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    # if len(image.shape) == 3:
    if dataset == 'Synapse':
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]    
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs, _ = net(input)
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
            outputs, _ = net(input)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()
            # Resize prediction back to original dimensions
            pred = zoom(out, (H/patch_size[0], W/patch_size[1]), order=0)
            prediction = pred  # Direct assignment to 2D array
    else:
        raise ValueError("Unknown dataset")
    
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

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
    # flops = FlopCountAnalysis(model, dummy_input)
    # total_flops = flops.total()

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
        # "flops(G)": total_flops / 1e9,
        "latency(ms)": latency_ms,
        "throughput(images/s)": throughput,
        "macs(G)": macs
    }

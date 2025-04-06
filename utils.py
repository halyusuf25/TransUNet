import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import time
from fvcore.nn import FlopCountAnalysis
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


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
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
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
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



# Function to calculate the number of parameters in the model
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params >= 1e9:  # If parameters are in billions
        return total_params / 1e9, 'B'
    elif total_params >= 1e6:  # If parameters are in millions
        return total_params / 1e6, 'M'
    else:  # If parameters are in thousands or less
        return total_params / 1e3, 'K'

def calculate_inference_latency(model, input_tensor, num_runs=500):
    model.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(20):
            _ = model(input_tensor)
        
        # Measure latency
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(input_tensor)
        end_time = time.time()
    
    avg_latency = (end_time - start_time) / num_runs
    return avg_latency


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


def measure_model_metrics(
    model: torch.nn.Module,
    input_channels: int = 3,
    input_height: int = 224,
    input_width: int = 224,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[float, float, float, float]:
    """
    Measure Params, Latency (in milliseconds), Throughput, and MACs for a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate (e.g., Vision Transformer).
        input_channels (int): Number of input channels (default: 3 for RGB).
        input_height (int): Height of the input image (default: 224).
        input_width (int): Width of the input image (default: 224).
        batch_size (int): Batch size for throughput measurement (default: 32).
        device (str): Device to run on ('cuda' or 'cpu', default: auto-detected).

    Returns:
        Tuple[float, float, float, float]: (params in millions, latency in milliseconds,
                                           throughput in images/second, macs in billions)
    """
    # Move model to the specified device and set to evaluation mode
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Create sample inputs: single image and a batch
    single_input = torch.randn(1, input_channels, input_height, input_width).to(device)
    batch_input = torch.randn(batch_size, input_channels, input_height, input_width).to(device)

    # --- Measure number of parameters ---
    params = sum(p.numel() for p in model.parameters()) / 1e6  # Convert to millions

    # --- Measure latency (time for a single forward pass in milliseconds) ---
    with torch.no_grad():
        model(single_input)  # Warm-up run

    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            model(single_input)
        end_event.record()
        torch.cuda.synchronize()
        latency = start_event.elapsed_time(end_event)  # Already in milliseconds
    else:
        start_time = time.time()
        with torch.no_grad():
            model(single_input)
        latency = (time.time() - start_time) * 1000  # Convert seconds to milliseconds

    # --- Measure throughput (images processed per second) ---
    with torch.no_grad():
        model(batch_input)  # Warm-up run

    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            model(batch_input)
        end_event.record()
        torch.cuda.synchronize()
        time_taken = start_event.elapsed_time(end_event) / 1000  # Convert to seconds for throughput
    else:
        start_time = time.time()
        with torch.no_grad():
            model(batch_input)
        time_taken = time.time() - start_time  # In seconds
    throughput = batch_size / time_taken

    # --- Measure MACs (Multiply-Accumulate Operations) ---
    try:
        macs, _ = profile(model, inputs=(single_input,), verbose=False)
        macs = macs / 1e9  # Convert to billions
    except Exception as e:
        print(f"Warning: Could not compute MACs due to {e}. Setting MACs to 0.")
        macs = 0.0

    # flops = FlopCountAnalysis(model, (single_input,))
    # print(f"FLOPs: {flops.total()}")
    # Print results
    print(f"Number of parameters: {params:.2f} M")
    print(f"Latency: {latency:.2f} ms")
    print(f"Throughput: {throughput:.2f} images/second")
    print(f"MACs: {macs:.2f} G")

    # return params, latency, throughput, macs



import torch.nn as nn

def vision_transformer_flops(model, input_size=(3, 224, 224), config=None, use_shsa=False):
    flops = 0
    C, H, W = input_size  # Initial input dimensions
    seq_len = H * W  # Will store transformer sequence length
    attn_dim = 768  # From your architecture description
    qk_dim = 16  # Fixed per SHViT design
    if config is not None:
        pdim = int(config.hidden_size / 4.67)  # Partial dimension (r = 1/4.67 from SHViT paper)
    else:
        pdim = int(768/4.67)  # Default value if config is not provided


    # Helper to calculate Conv2D output dimensions
    def conv2d_output(h, w, conv):
        return (
            (h + 2*conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1,
            (w + 2*conv.padding[1] - conv.kernel_size[1]) // conv.stride[1] + 1
        )

    # Process ResNetV2 backbone
    def process_resnet_block(block):
        nonlocal H, W, flops
        for unit in block.children():
            # Process each PreActBottleneck unit
            for name, layer in unit.named_children():
                if isinstance(layer, nn.Conv2d):
                    # Conv1 (1x1), Conv2 (3x3), Conv3 (1x1)
                    h, w = conv2d_output(H, W, layer)
                    flops += (
                        layer.in_channels * layer.out_channels *
                        layer.kernel_size[0] * layer.kernel_size[1] *
                        h * w * 2  # 2 for MAC
                    )
                    H, W = h, w
                    
                # Handle downsampling
                if 'downsample' in name and isinstance(layer, nn.Conv2d):
                    h, w = conv2d_output(H, W, layer)
                    flops += (
                        layer.in_channels * layer.out_channels *
                        layer.kernel_size[0] * layer.kernel_size[1] *
                        h * w * 2
                    )

    # Process main components
    for name, module in model.named_modules():
        # ResNetV2 backbone
        if 'hybrid_model' in name:
            if isinstance(module, nn.Conv2d):
                H, W = conv2d_output(H, W, module)
                flops += (
                    module.in_channels * module.out_channels *
                    module.kernel_size[0] * module.kernel_size[1] *
                    H * W * 2
                )
                
            elif 'block' in name and isinstance(module, nn.Sequential):
                process_resnet_block(module)

        # Patch embeddings
        if 'patch_embeddings' in name and isinstance(module, nn.Conv2d):
            H, W = conv2d_output(H, W, module)
            flops += (
                module.in_channels * module.out_channels *
                module.kernel_size[0] * module.kernel_size[1] *
                H * W * 2
            )
            seq_len = H * W  # Sequence length for transformer

        # Transformer attention layers
        if isinstance(module, nn.Linear) and 'attn' in name:
            # Q/K/V projections
            flops += 2 * module.in_features * module.out_features

        # MLP layers
        if isinstance(module, nn.Linear) and 'ffn' in name:
            flops += 2 * module.in_features * module.out_features

        # # Attention matrix operations
        # if isinstance(module, Attention):
        #     # Q@K^T: [B, H, N, D] @ [B, H, D, N] -> [B, H, N, N]
        #     flops += 2 * seq_len * seq_len * attn_dim
        #     # Attn@V: [B, H, N, N] @ [B, H, N, D] -> [B, H, N, D]
        #     flops += 2 * seq_len * attn_dim * seq_len
                # Attention matrix operations
        if use_shsa:
            # SHViT-specific calculation
            flops += 2 * seq_len * seq_len * qk_dim    # Q@K^T
            flops += 2 * seq_len * pdim * seq_len       # Attn@V
            # Projection from pdim to (qk_dim*2 + pdim)
            flops += 2 * pdim * (2*qk_dim + pdim) * seq_len
        else:
            # Original MHSA calculation
            flops += 2 * seq_len * seq_len * attn_dim  # Q@K^T
            flops += 2 * seq_len * attn_dim * seq_len  # Attn@V

        # Decoder convolutions
        if 'decoder' in name and isinstance(module, nn.Conv2d):
            h, w = conv2d_output(H, W, module)
            flops += (
                module.in_channels * module.out_channels *
                module.kernel_size[0] * module.kernel_size[1] *
                h * w * 2
            )
            H, W = h, w

    print(f"Total FLOPs: {flops / 1e9:.2f} G")
    
    

def distillation_loss(student_logits, teacher_logits):
    """
    Computes pixel-wise distillation loss using KL divergence between teacher and student logits.
    
    Args:
        student_logits (torch.Tensor): Student network logits, shape (B, C, H, W)
        teacher_logits (torch.Tensor): Teacher network logits, shape (B, C, H, W)
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    temperature = 2.0  # Temperature for softening logits
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    
    # KLDivLoss(reduction='none') returns (B, C, H, W)
    loss = nn.KLDivLoss(reduction='none')(student_log_probs, teacher_probs.detach())
    
    # Sum over classes (dim=1), then average over H, W, and batch
    loss_per_pixel = loss.sum(dim=1)  # Shape: (B, H, W)
    loss_per_sample = loss_per_pixel.mean(dim=(1, 2))  # Shape: (B,)
    total_loss = loss_per_sample.mean()  # Scalar
    
    return total_loss

def pairwise_distillation_loss(student_features, teacher_features):
    """
    Computes pairwise distillation loss based on similarity matrices of feature maps.
    
    Args:
        student_features (torch.Tensor): Student feature maps, shape (B, C, H, W)
        teacher_features (torch.Tensor): Teacher feature maps, shape (B, C, H, W)
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    B, C, H, W = student_features[0].shape
    L = H * W  # Total spatial locations

    # Reshape and L2-normalize features along the channel dimension
    student_f = F.normalize(student_features.view(B, C, -1), p=2, dim=1)  # (B, C, L)
    teacher_f = F.normalize(teacher_features.view(B, C, -1), p=2, dim=1)  # (B, C, L)

    # Compute similarity matrices via batch matrix multiplication
    student_sim = torch.bmm(student_f.transpose(1, 2), student_f)  # (B, L, L)
    teacher_sim = torch.bmm(teacher_f.transpose(1, 2), teacher_f)  # (B, L, L)

    # MSE loss (mean over B, L, L)
    loss = F.mse_loss(student_sim, teacher_sim.detach())

    return loss


def dist_loss(student_logits, teacher_logits, student_features, teacher_features):
    temperature = 2.0  # Temperature for softening logits
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    
    # KLDivLoss(reduction='none') returns (B, C, H, W)
    loss = nn.KLDivLoss(reduction='none')(student_log_probs, teacher_probs.detach())
    
    # Sum over classes (dim=1), then average over H, W, and batch
    loss_per_pixel = loss.sum(dim=1)  # Shape: (B, H, W)
    loss_per_sample = loss_per_pixel.mean(dim=(1, 2))  # Shape: (B,)
    total_resp_loss = loss_per_sample.mean()  # Scalar

    B, C, H, W = student_features[0].shape
    # print(f"Student features shape: {student_features[0].shape}")
    # print(f"length of student features: {len(student_features)}")
    # print(f"length of teacher features: {len(teacher_features)}")
    # print(f"Teacher features shape: {teacher_features[0].shape}")
    number_layers = len(student_features)
    L = H * W  # Total spatial locations
    # student_features = student_features[-1]
    # teacher_features = teacher_features[-1]
    st_ft = student_features
    tr_ft = teacher_features
    # Reshape and L2-normalize features along the channel dimension
    pairw_loss = 0.0
    for l in range(number_layers):
        student_features = st_ft[l]
        teacher_features = tr_ft[l]
        student_f = F.normalize(student_features.view(B, C, -1), p=2, dim=1)  # (B, C, L)
        teacher_f = F.normalize(teacher_features.view(B, C, -1), p=2, dim=1)  # (B, C, L)

        # Compute similarity matrices via batch matrix multiplication
        student_sim = torch.bmm(student_f.transpose(1, 2), student_f)  # (B, L, L)
        teacher_sim = torch.bmm(teacher_f.transpose(1, 2), teacher_f)  # (B, L, L)

        # MSE loss (mean over B, L, L)
        pairw_loss += F.mse_loss(student_sim, teacher_sim.detach())


    total_kd_loss = 0.5 * total_resp_loss + 0.5 * (pairw_loss / number_layers)

    return total_kd_loss


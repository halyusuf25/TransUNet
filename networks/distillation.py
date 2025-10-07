import torch
import torch.nn.functional as F
import torch.nn as nn

def distillation_loss(student_logits, teacher_logits):
    """
    Computes pixel-wise distillation loss using KL divergence between teacher and student logits.
    
    Args:
        student_logits (torch.Tensor): Student network logits, shape (B, C, H, W)
        teacher_logits (torch.Tensor): Teacher network logits, shape (B, C, H, W)
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    temperature = 3.0  # Temperature for softening logits
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
    temperature = 3.0  # Temperature for softening logits
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


    total_kd_loss = 0.5 * total_resp_loss + 0.5 * pairw_loss

    return total_kd_loss
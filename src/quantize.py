
import torch
import torch.nn.functional as F


def inc_awq_image_to_nchw(image, args):
    """Convert one calibration sample from repo dataset format to model-ready NCHW tensor."""
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)

    image = image.detach().cpu().float()
    img_size = int(args.img_size)

    if args.dataset in ["Synapse", "ACDC"]:
        # DataLoader usually gives [1, D, H, W] for volume tests.
        # Convert to [D, 1, H, W], so each slice becomes one calibration image.
        if image.dim() == 4 and image.shape[0] == 1:
            x = image.squeeze(0).unsqueeze(1)
        elif image.dim() == 3:
            x = image.unsqueeze(1)
        elif image.dim() == 4 and image.shape[1] in (1, 3):
            x = image
        else:
            raise ValueError(
                f"Unsupported {args.dataset} calibration image shape: {tuple(image.shape)}"
            )
    else:
        # Frame datasets: accept [1, H, W, 3], [1, 3, H, W], [H, W, 3], [3, H, W].
        if image.dim() == 4 and image.shape[1] in (1, 3):
            x = image
        elif image.dim() == 4 and image.shape[-1] in (1, 3):
            x = image.permute(0, 3, 1, 2)
        elif image.dim() == 3 and image.shape[0] in (1, 3):
            x = image.unsqueeze(0)
        elif image.dim() == 3 and image.shape[-1] in (1, 3):
            x = image.permute(2, 0, 1).unsqueeze(0)
        elif image.dim() == 2:
            x = image.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported calibration image shape: {tuple(image.shape)}")

    if x.shape[-2:] != (img_size, img_size):
        x = F.interpolate(
            x,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )

    return x.contiguous()


def collect_inc_awq_calib_inputs(args, calib_loader, max_forwards, chunk_size=8):
    """Collect a small list of tensors to run through INC AWQ calibration."""
    calib_inputs = []
    max_forwards = int(max_forwards)
    chunk_size = max(1, int(chunk_size))

    for sampled_batch in calib_loader:
        x = inc_awq_image_to_nchw(sampled_batch["image"], args)

        for chunk in x.split(chunk_size, dim=0):
            calib_inputs.append(chunk)
            if len(calib_inputs) >= max_forwards:
                return calib_inputs

    return calib_inputs
import base64
import io
from typing import List

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """
    Convert a tensor to a PIL Image.
    """
    shape = tensor.shape
    assert len(shape) == 4, "Tensor must have 4 dimensions"
    bsz = shape[0]
    img_cpu = tensor.cpu().numpy()
    images = []
    for i in range(bsz):
        if img_cpu[i].max() <= 1.:
            img_cpu[i] = (img_cpu[i] * 255).astype(np.uint8)
        else:
            img_cpu[i] = (img_cpu[i]).astype(np.uint8)
        images.append(Image.fromarray(img_cpu[i]))
    return images

def pil_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    tensors = []
    for img in images:
        np_img = np.array(img)
        if np_img.ndim == 2:  # grayscale to (H, W)
            np_img = np_img[:, :, None]
        np_img = np_img.astype(np.float32) / 255.0
        # Ensure 3 channels (RGB)
        if np_img.shape[2] == 1:  # grayscale, replicate to RGB
            np_img = np.repeat(np_img, 3, axis=2)
        elif np_img.shape[2] == 4:  # RGBA, drop alpha
            np_img = np_img[:, :, :3]
        elif np_img.shape[2] == 3:
            pass  # already RGB
        else:
            raise ValueError(f"Unsupported number of channels: {np_img.shape[2]}")
        tensors.append(torch.from_numpy(np_img))
    batch = torch.stack(tensors, dim=0)  # (B, H, W, C)
    return batch

def pil_to_base64(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_tensor(base64_image: str) -> torch.Tensor:
    image_data = base64.b64decode(base64_image)
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGBA")
    image_array = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array)[None]
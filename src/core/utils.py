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


def pil_to_base64(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_tensor(base64_image: str) -> torch.Tensor:
    image_data = base64.b64decode(base64_image)
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGBA")
    image_array = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array)[None]
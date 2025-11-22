import asyncio
import io
import os

from google.genai import types
from google.genai.types import Image as GenaiImage
from google.genai.types import Part
from PIL import Image

try:
    from ..core.gemini_client import *
    from ..core.utils import pil_to_tensor, tensor_to_pil
except ImportError:
    from core.gemini_client import *
    from core.utils import pil_to_tensor, tensor_to_pil


class NanoBananaNode:
    def __init__(self):
        self.client = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (NANO_BANANA_MODELS,),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "prompt",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            },
            "optional": {
                "system_instruction": ("STRING", {
                    "multiline": True,
                }),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    FUNCTION = "generate"

    CATEGORY = "Even"

    async def generate(self, model, prompt, temperature, system_instruction=None, image_1=None, image_2=None, image_3=None, image_4=None, audio=None, video=None, files=None):
        if self.client is None:
            self.client = create_gemini_client()

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        ) if system_instruction else None

        contents = [prompt]
        imgs = []
        for img in [image_1, image_2, image_3, image_4]:
            if img is not None:
                imgs.append(tensor_to_pil(img))
        for pil_img in imgs:
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            contents.append(Part.from_bytes(data=img_bytes, mime_type="image/png"))

        response = await self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        text = ""
        output_image = []
        for part in response.parts:
            if part.text is not None:
                text += part.text
            elif part.inline_data is not None:
                genai_image: GenaiImage = part.as_image()
                image_bytes = genai_image.image_bytes
                image = Image.open(io.BytesIO(image_bytes))
                output_image.append(image)
        return (text, pil_to_tensor(output_image))

# A dictionary that ComfyUI uses to register the nodes in this file
IMPL_NODE_CLASS_MAPPINGS = {
    "NanoBanana": NanoBananaNode
}

# A dictionary that ComfyUI uses to display the node names in the UI
IMPL_NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBanana": "NanoBanana"
}

if __name__ == "__main__":
    import asyncio
    asyncio.run(NanoBananaNode().generate(api_key=os.environ.get("GEMINI_API_KEY"), model="gemini-3-pro-image-preview", prompt="生成一张沙滩的图片", temperature=0.5))
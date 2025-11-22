import asyncio
import io
import os

from google.genai import types
from google.genai.types import Part

try:
    from ..core.gemini_client import *
    from ..core.utils import tensor_to_pil
except ImportError:
    from core.gemini_client import *
    from core.utils import tensor_to_pil

class GeminiNode:
    def __init__(self):
        self.client = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (GEMINI_MODELS,),
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
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "video": ("VIDEO",),
                "files": ("FILE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"

    CATEGORY = "Even"

    async def generate(self, model, prompt, temperature, system_instruction=None, images=None, audio=None, video=None, files=None):
        if self.client is None:
            self.client = create_gemini_client()

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        ) if system_instruction else None

        contents = [prompt]
        imgs = []
        if images is not None:
            imgs = tensor_to_pil(images)
        for pil_img in imgs:
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            contents.append(Part.from_bytes(data=img_bytes, mime_type="image/png"))

        response = await self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        return (response.text,)

# A dictionary that ComfyUI uses to register the nodes in this file
IMPL_NODE_CLASS_MAPPINGS = {
    "Gemini": GeminiNode
}

# A dictionary that ComfyUI uses to display the node names in the UI
IMPL_NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini": "Gemini"
}

if __name__ == "__main__":
    import asyncio
    asyncio.run(GeminiNode().generate(model="gemini-3-pro-preview", prompt="Hello, world!", temperature=0.5))
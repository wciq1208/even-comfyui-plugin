# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import io
import os

from google import genai
from google.genai import types
from google.genai.types import Part

from ..core.gemini_client import *
from ..core.utils import tensor_to_pil


class GeminiNode:
    """
    A ComfyUI node for interacting with the Google Gemini models.

    This node allows users to send text prompts and optional images to the Gemini API
    and receive a generated text response. It supports various Gemini models and
    can be configured with a system instruction.
    """
    def __init__(self):
        """
        Initializes the node by setting the client to None.
        The client will be created on the first execution.
        """
        self.client = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": os.environ.get("GEMINI_API_KEY")
                }),
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
    FUNCTION = "generate_text"

    CATEGORY = "Even"

    async def generate_text(self, api_key, model, prompt, temperature, system_instruction=None, images=None, audio=None, video=None, files=None):
        if self.client is None:
            self.client = create_gemini_client(api_key)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        ) if system_instruction else None

        contents = [prompt]
        imgs = tensor_to_pil(images)
        for pil_img in imgs:
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            contents.append(Part.from_bytes(data=img_bytes, mime_type="image/png"))

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=model,
            contents=contents,
            config=config,
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
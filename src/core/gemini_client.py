from google import genai


def create_gemini_client():
    client = genai.Client(
        vertexai=True).aio
    return client


GEMINI_MODELS = [
                    'gemini-3-pro-preview',
                    'gemini-2.5-flash',
                    'gemini-2.5-pro',
                    'gemini-2.5-flash-lite'
                ]

NANO_BANANA_MODELS = [
    'gemini-3-pro-image-preview',
    'gemini-2.5-flash-image',
]
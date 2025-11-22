from google import genai


def create_gemini_client(api_key: str):
    client = genai.Client(
        api_key=api_key,
        vertexai=True,
        api_version="v1",)
    return client


GEMINI_MODELS = [
                    'gemini-3-pro-preview',
                    'gemini-2.5-flash',
                    'gemini-2.5-pro',
                    'gemini-2.5-flash-lite'
                ]
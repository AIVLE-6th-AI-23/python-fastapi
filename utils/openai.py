from openai import OpenAI
from .constants import OPENAI_API_KEY

def load_openai_client() :
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://api.perplexity.ai"
    )
    return client
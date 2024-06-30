import requests
from retry import retry


def initiate_embedding(content, hf_api_key):
    """HuggingFace API for getting embeddings -- somewhat based on https://huggingface.co/blog/getting-started-with-embeddings"""
    model_id = "sentence-transformers/all-MiniLM-L6-v2"  # this model only embedds the first 256 words, but that is enough for our purposes and it is a small load which is better
    hf_token = hf_api_key
    content = content
    api_url = (
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    )
    hf_headers = {"Authorization": f"Bearer {hf_token}"}
    # the initial API call loads the model, thus it has to use Retry decorator while the model loads, subsequent API calls will run faster
    @retry(tries=10, delay=10)
    def embedding(content):
        response = requests.post(api_url, headers=hf_headers, json={"inputs": content})
        embedding = response.json()
        if isinstance(embedding, list):
            return embedding
        elif list(embedding.keys())[0] == "error":
            raise RuntimeError(
                "The model is currently loading, please re-run the query."
            )

    embedding = embedding(content)
    # print("got embedding")
    return embedding

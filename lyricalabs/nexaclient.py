import requests
from typing import Optional, Dict, Any

class NexaClient:
    """
    XanaxWay Nexa API istemcisi
    """

    BASE_URL = "https://api.xanaxway.com/v4/llm/nexa/generative/model/completions"

    def __init__(self, token: str):
        self.token = token

    def generate(
        self,
        prompt: str,
        model: str = "nexa-7.0-express",
        temperature: float = 0.9,
        max_tokens: int = 4098,
        top_p: float = 0.95,
        frequency_penalty: float = 0.2,
        presence_penalty: float = 0.1,
        custom_system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Nexa modelinden yanıt üretir.
        """
        
        headers = {
            "X-API-Key": self.token,
            "Content-Type": "application/json"
        }

        payload = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "custom_system_instruction": custom_system_instruction
        }

        response = requests.post(self.BASE_URL, json=payload, headers=headers)
        try:
            data = response.json()
        except Exception:
            return {
                "status": "Error",
                "message": "API yanıtı okunamadı",
                "raw_response": response.text
            }

        return data

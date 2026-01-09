import requests
from typing import List, Dict, Any, Optional, Generator

class NexaClient:
    """Lyrica Labs Nexa LLM API istemci sınıfı"""
    
    BASE_URL = "https://api-lyricalabs.vercel.app/v4/llm/nexa/generative/model/completions"
     
    # Güncellenmiş model listesi -
    MODELS = [
        "nexa-5.0-preview",
        "nexa-3.7-pro", 
        "nexa-5.0-intimate",
        "nexa-6.1-infinity",
        "nexa-7.0-insomnia",  # Yeni eklenen model
        "nexa-6.1-code-llm",
        "nexa-7.0-express",
        "gpt-5-mini-chatgpt"
    ]
    
    MODEL_DESCRIPTIONS = {
    "nexa-5.0-preview": "Nexa 5.0 Preview, geniş bağlam anlayışı ve mantıksal akışıyla daha derin ve doğal sohbet deneyimleri sunan gelişmiş bir dil modelidir.",
    "nexa-3.7-pro": "Nexa 3.7 Pro, geniş bağlam anlayışı ve akıcı mantığıyla keyifli sohbetler için tasarlanmış bir dil modelidir.",
    "nexa-5.0-intimate": "Nexa 5.0 Intimate, geniş bağlam anlayışı ve akıcı mantığıyla daha samimi ve kişisel sohbetler için tasarlanmış bir dil modelidir.",
    "nexa-6.1-infinity": "Ultra geniş bağlam penceresi ve insan düzeyinde mantıksal kurgulama kapasitesi ile LLM dünyasının en güçlü dil modeli.",
    "nexa-7.0-insomnia": "**Nexa 7.0 Insomnia**\nDuygusal bağlamı ve **insan düşünce akışını** merkeze alan _Nexa 7.0 Insomnia_, doğal dilde empati kurabilen **en gelişmiş modelimizdir**. Yapay “terapist” kalıplarından uzak durarak, *gerçek bir insan gibi düşünen* ve cevaplayan bir deneyim sunar.",
    "nexa-6.1-code-llm": "Nexa 6.1 Code, ultra geniş bağlam ve insan düzeyinde mantıkla kodlama odaklı AI dünyasının en güçlü dil modeli.",
    "nexa-7.0-express": "**Nexa 7.0 Express**, yüksek uyumluluk ve görev odaklı çalışacak şekilde tasarlanmış, kurumsal kullanıma uygun gelişmiş bir geniş dil modelidir **(LLM)**.\nHızlı, tutarlı ve net çıktılar üretir.",
    "gpt-5-mini-chatgpt": "ChatGPT 5 Mini, OpenAI tarafından geliştirilen ve Lyrica Labs tarafından sunulan, kompakt ve akıcı sohbet deneyimi sunan bir dil modelidir."
    }

    def __init__(self, token: str, base_url: Optional[str] = None):
        """
        NexaClient başlatıcı
        
        Args:
            token: API token
            base_url: Özel API endpoint (opsiyonel)
        """
        self.token = token
        self.base_url = base_url or self.BASE_URL
        
    def list_models(self, with_descriptions: bool = False) -> List[str] | Dict[str, str]:
        """
        Mevcut modelleri listeler
        
        Args:
            with_descriptions: Açıklamalarla birlikte döndürür
            
        Returns:
            Model listesi veya açıklamalı sözlük
        """
        if with_descriptions:
            return self.MODEL_DESCRIPTIONS.copy()
        return self.MODELS.copy()
    
    def get_model_info(self, model: str) -> Dict[str, str]:
        """
        Belirli bir model hakkında bilgi döndürür
        
        Args:
            model: Model adı
            
        Returns:
            Model bilgileri
            
        Raises:
            ValueError: Model bulunamazsa
        """
        if model not in self.MODEL_DESCRIPTIONS:
            raise ValueError(f"Model '{model}' bulunamadı")
        
        return {
            "name": model,
            "description": self.MODEL_DESCRIPTIONS[model],
            "category": self._get_model_category(model)
        }
    
    def _get_model_category(self, model: str) -> str:
        """Model kategorisini belirler"""
        categories = {
            "code": ["code-llm"],
            "creative": ["intimate"],
            "fast": ["express"],
            "general": ["preview", "pro", "infinity", "insomnia"],
            "compatible": ["chatgpt"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in model for keyword in keywords):
                return category
        return "general"
    
    def generate_text(
        self, 
        prompt: str, 
        model: str = "nexa-5.0-preview",
        temperature: float = 0.6, 
        max_tokens: int = 4096,
        top_p: float = 0.95, 
        frequency_penalty: float = 0.2,
        presence_penalty: float = 0.1, 
        custom_system_instruction: str = "",
        stream: bool = False,
        timeout: int = 30
    ) -> Dict[str, Any] | Generator[str, None, None]:
        """
        Nexa API ile metin üretir
        
        Args:
            prompt: Kullanıcı girişi
            model: Kullanılacak model
            temperature: Yaratıcılık seviyesi (0.0-2.0)
            max_tokens: Maksimum token sayısı
            top_p: Çeşitlilik kontrolü
            frequency_penalty: Tekrar cezası
            presence_penalty: Yeni konu ödülü
            custom_system_instruction: Özel sistem talimatı
            stream: Stream modu
            timeout: İstek zaman aşımı (saniye)
            
        Returns:
            API yanıtı veya stream generator
            
        Raises:
            ValueError: Geçersiz model veya parametreler
            requests.exceptions.RequestException: API hatası
        """
        if model not in self.MODELS:
            raise ValueError(f"Model '{model}' bulunamadı! Mevcut modeller: {self.MODELS}")
        
        if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise ValueError("temperature 0 ile 2 arasında olmalıdır")
        
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens pozitif bir tam sayı olmalıdır")
        
        payload = {
            "token": self.token,
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "custom_system_instruction": custom_system_instruction,
            "stream": stream
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"LyricalabsNexaClient/0.3.0"
        }
        
        try:
            response = requests.post(
                self.base_url, 
                json=payload, 
                headers=headers,
                timeout=timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_stream_response(response)
            else:
                return response.json()
                
        except requests.exceptions.Timeout:
            raise Exception("API yanıt vermedi. Lütfen daha sonra tekrar deneyin.")
        except requests.exceptions.ConnectionError:
            raise Exception("API'ye bağlanılamadı. İnternet bağlantınızı kontrol edin.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise Exception("Geçersiz API token. Lütfen token'ınızı kontrol edin.")
            elif e.response.status_code == 429:
                raise Exception("Rate limit aşıldı. Lütfen bir süre bekleyin.")
            elif e.response.status_code == 500:
                raise Exception("Sunucu hatası. Lütfen daha sonra tekrar deneyin.")
            else:
                raise Exception(f"API hatası: {e.response.status_code} - {e.response.text}")
    
    def _handle_stream_response(self, response) -> Generator[str, None, None]:
        """Stream yanıtını işler"""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    yield decoded_line[6:]  # "data: " kısmını çıkar
    
    def health_check(self) -> Dict[str, Any]:
        """
        API bağlantısını ve token geçerliliğini kontrol eder
        
        Returns:
            Sağlık durumu bilgileri
        """
        try:
            # Test isteği yap
            test_response = self.generate_text(
                prompt="test",
                model="nexa-5.0-preview",
                max_tokens=1,
                temperature=0
            )
            
            return {
                "status": "healthy",
                "api_accessible": True,
                "token_valid": True,
                "models_available": len(self.MODELS)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "api_accessible": False,
                "token_valid": False,
                "error": str(e)
            }
    
    def batch_generate(
        self,
        prompts: List[str],
        model: str = "nexa-5.0-preview",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Birden fazla prompt için toplu üretim yapar
        
        Args:
            prompts: Prompt listesi
            model: Kullanılacak model
            **kwargs: Diğer generate_text parametreleri
            
        Returns:
            Yanıt listesi
        """
        results = []
        for prompt in prompts:
            try:
                result = self.generate_text(prompt=prompt, model=model, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "prompt": prompt})
        
        return results


def create_client(token: str, base_url: Optional[str] = None) -> NexaClient:
    """Hızlı client oluşturma yardımcı fonksiyonu"""
    return NexaClient(token=token, base_url=base_url)

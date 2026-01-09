import requests
from typing import List, Dict, Any, Optional, Generator
import json

class NexaClient:
    """Lyrica Labs Nexa LLM API istemci sınıfı"""
    
    BASE_URL = "https://api-lyricalabs.vercel.app/v4/llm/nexa/generative/model/completions"
    
    # Güncellenmiş model listesi
    MODELS = [
        "nexa-5.0-preview",
        "nexa-3.7-pro", 
        "nexa-5.0-intimate",
        "nexa-6.1-infinity",
        "nexa-7.0-insomnia",
        "nexa-6.1-code-llm",
        "nexa-7.0-express",
        "gpt-5-mini-chatgpt"
    ]
    
    MODEL_DESCRIPTIONS = {
        "nexa-5.0-preview": "Genel amaçlı, dengeli model",
        "nexa-3.7-pro": "İş odaklı, profesyonel çıktılar",
        "nexa-5.0-intimate": "Yaratıcı yazım ve duygusal içerik",
        "nexa-6.1-infinity": "Büyük bağlam, detaylı analiz",
        "nexa-7.0-insomnia": "24/7 optimize edilmiş, yüksek performans, empati yeteneği",
        "nexa-6.1-code-llm": "Kod yazma ve analiz için özel",
        "nexa-7.0-express": "Hızlı yanıt, düşük gecikme",
        "gpt-5-mini-chatgpt": "ChatGPT uyumlu mini model"
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
    ) -> Dict[str, Any]:
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
            API yanıtı
            
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
            "User-Agent": f"LyricalabsNexaClient/0.3.1"
        }
        
        try:
            response = requests.post(
                self.base_url, 
                json=payload, 
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            
            # API yanıtını parse et
            result = response.json()
            
            # Eski formatla uyumluluk için dönüştür
            if self._is_new_format(result):
                return self._convert_to_old_format(result)
            
            return result
                
        except requests.exceptions.Timeout:
            raise Exception("API yanıt vermedi. Lütfen daha sonra tekrar deneyin.")
        except requests.exceptions.ConnectionError:
            raise Exception("API'ye bağlanılamadı. İnternet bağlantınızı kontrol edin.")
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Hatası: {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg = error_data.get("mesaj", error_data.get("message", error_msg))
            except:
                pass
            
            if e.response.status_code == 401:
                raise Exception(f"Geçersiz API token. {error_msg}")
            elif e.response.status_code == 429:
                raise Exception(f"Rate limit aşıldı. {error_msg}")
            elif e.response.status_code == 500:
                raise Exception(f"Sunucu hatası. {error_msg}")
            else:
                raise Exception(f"API hatası: {error_msg}")
        except json.JSONDecodeError as e:
            raise Exception(f"API yanıtı geçersiz JSON formatında: {e}")
    
    def _is_new_format(self, response: Dict[str, Any]) -> bool:
        """Yeni API formatını kontrol et"""
        return "cikti" in response and "basarilimi" in response
    
    def _convert_to_old_format(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Yeni formatı eski format'a dönüştür"""
        if not response.get("basarilimi", False):
            raise Exception(f"API hatası: {response.get('mesaj', 'Bilinmeyen hata')}")
        
        # Eski OpenAI benzeri format'a dönüştür
        converted = {
            "id": f"chatcmpl-{hash(str(response))}",
            "object": "chat.completion",
            "created": 0,  # Timestamp bilgisi yok
            "model": response.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.get("cikti", "")
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            # Orijinal yanıtı da ekle
            "original_response": response
        }
        
        return converted
    
    def generate_text_simple(
        self,
        prompt: str,
        model: str = "nexa-5.0-preview",
        **kwargs
    ) -> str:
        """
        Basit metin üretimi - sadece metin döndürür
        
        Args:
            prompt: Kullanıcı girişi
            model: Kullanılacak model
            **kwargs: Diğer parametreler
            
        Returns:
            Üretilen metin
        """
        response = self.generate_text(prompt=prompt, model=model, **kwargs)
        
        # Yeni format
        if "original_response" in response:
            return response["original_response"]["cikti"]
        # Eski format
        elif "choices" in response:
            return response["choices"][0]["message"]["content"]
        else:
            return str(response)
    
    def get_response_details(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        API yanıtından detaylı bilgileri çıkarır
        
        Args:
            response: generate_text()'ten dönen yanıt
            
        Returns:
            Detaylı bilgiler
        """
        if "original_response" in response:
            orig = response["original_response"]
            return {
                "success": orig.get("basarilimi", False),
                "message": orig.get("mesaj", ""),
                "output": orig.get("cikti", ""),
                "model": orig.get("model", ""),
                "warnings": orig.get("guvenlik_uyarilari", []),
                "parameters": orig.get("kullanilan_parametreler", {}),
                "info": orig.get("bilgi", ""),
                "web_search": orig.get("web_arama_kullanildi", False),
                "links": orig.get("info", {})
            }
        else:
            return {
                "success": True,
                "output": response["choices"][0]["message"]["content"],
                "model": response.get("model", ""),
                "warnings": [],
                "parameters": {}
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        API bağlantısını ve token geçerliliğini kontrol eder
        
        Returns:
            Sağlık durumu bilgileri
        """
        try:
            # Basit bir test isteği yap
            response = self.generate_text(
                prompt="test",
                model="nexa-5.0-preview",
                max_tokens=1,
                temperature=0
            )
            
            details = self.get_response_details(response)
            
            return {
                "status": "healthy",
                "api_accessible": True,
                "token_valid": details["success"],
                "models_available": len(self.MODELS),
                "details": details
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

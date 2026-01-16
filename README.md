# Lyricalabs Nexa Python Kütüphanesi

Lyricalabs Nexa, **Lyrica Labs** tarafından geliştirilen geniş veri LLM modellerine erişim sağlayan Python kütüphanesidir. Bu kütüphane ile **Nexa modellerini kolayca kullanabilir** ve metin üretimi, kod analizi gibi senaryolarda hızlıca entegre edebilirsiniz.

---

## 📦 Kurulum

```bash
pip install lyricalabs

```
---

🔑 API Token Alma

Kütüphaneyi kullanmak için **API** token’a ihtiyacınız var:

1. [Lyricalabs Platform](https://lyricalabs.com) adresine girin


2. Kayıt olun ve giriş yapın


3. Dashboard’dan [API](https://lyricalabs.com/lyrica-labs-apis) token’ınızı alın




---

🚀 Hızlı Başlangıç
```python
from lyricalabs import NexaClient

# API token'ınız ile client oluşturun
client = NexaClient(token="API_TOKENİNİZ")

# Örnek prompt ve parametreler
prompt = "Python'da yapay zeka uygulamaları nasıl geliştirilir?"

response = client.generate(
    prompt=prompt,
    model="nexa-7.0-express",      # Hızlı yanıt modeli
    temperature=0.6,               # Yaratıcılık seviyesi
    max_tokens=500,                # Üretilecek maksimum token
    top_p=0.95,                    # Çeşitlilik kontrolü
    frequency_penalty=0.2,         # Tekrar cezası
    presence_penalty=0.1,          # Yeni konu ödülü
    custom_system_instruction="Cevapları Türkçe ve samimi ver."  # Opsiyonel sistem talimatı
)

if response.get("basarilimi"):
    print("✅ Yanıt:\n", response.get("output"))
else:
    print("❌ Hata oluştu:")
    print(response.get("message"))
    print("Raw response:", response.get("raw_response"))
```

---

📚 Mevcut Modeller

| Model | Açıklama | Önerilen Kullanım |
| :--- | :--- | :--- |
| nexa-5.0-preview | Genel amaçlı, dengeli model | Her türlü metin üretimi |
| nexa-3.7-pro | İş odaklı, profesyonel çıktılar | Rapor, e-posta, belge |
| nexa-6.1-infinity | Büyük bağlam, detaylı analiz | Uzun form içerik, analiz |
| nexa-7.0-insomnia | Empati ve insan anlama kapasitesi | Duygusal içerik, destek sistemi |
| nexa-5.0-intimate | Yaratıcı yazım ve duygusal içerik | Hikaye, şiir, yaratıcı yazı |
| nexa-6.1-code-llm | Kod yazma ve analiz | Programlama, kod analizi |
| nexa-7.0-express | Hızlı yanıt, düşük gecikme | Chat, hızlı yanıt |
| gpt-5-mini-chatgpt | ChatGPT uyumlu mini model | ChatGPT benzeri uygulamalar |



---

⚙️ Parametreler

## 📚 Mevcut Modeller

| Model | Açıklama | Önerilen Kullanım |
|-------|----------|-----------------|
| nexa-5.0-preview | Genel amaçlı, dengeli model | Her türlü metin üretimi |
| nexa-3.7-pro | İş odaklı, profesyonel çıktılar | Rapor, e-posta, belge |
| nexa-6.1-infinity | Büyük bağlam, detaylı analiz | Uzun form içerik, analiz |
| nexa-7.0-insomnia | Empati ve insan anlama kapasitesi | Duygusal içerik, destek sistemi |
| nexa-5.0-intimate | Yaratıcı yazım ve duygusal içerik | Hikaye, şiir, yaratıcı yazı |
| nexa-6.1-code-llm | Kod yazma ve analiz | Programlama, kod analizi |
| nexa-7.0-express | Hızlı yanıt, düşük gecikme | Chat, hızlı yanıt |
| gpt-5-mini-chatgpt | ChatGPT uyumlu mini model | ChatGPT benzeri uygulamalar |

---
"""

---

🔍 Model Bilgisi Alma
```python
# Tüm modelleri açıklamalarıyla listeleyin
models = client.list_models(with_descriptions=True)
for model, desc in models.items():
    print(f"{model}: {desc}")

# Belirli bir model hakkında detaylı bilgi
model_info = client.get_model_info("nexa-7.0-insomnia")
print(f"""
Model: {model_info['name']}
Açıklama: {model_info['description']}
Kategori: {model_info['category']}
""")
```

---

🩺 Sistem Sağlık Kontrolü

```python
health = client.health_check()
if health.get("status") == "healthy":
    print("✅ API bağlantısı başarılı!")
    print(f"📊 Mevcut model sayısı: {health.get('models_available')}")
else:
    print("❌ API bağlantısı sorunlu:", health.get("error"))
```

---

❓ Sık Sorulan Sorular

1. API token’ımı nasıl alırım?
Lyricalabs platformundan kayıt olun ve dashboard’dan token oluşturun.


2. Hangi modeli kullanmalıyım?

Genel kullanım: nexa-5.0-preview

Duygusal içerik: nexa-7.0-insomnia

Kod yazma: nexa-6.1-code-llm

Hızlı yanıt: nexa-7.0-express



3. Rate limit var mı?
- Evet, token tipine göre değişir. Dashboard’dan kontrol edin.




---

📞 Destek ve İletişim

Website: https://lyricalabs.com

Nexa API Docs: https://docs.lyricalabs.com

Email: lyricalabs@gmail.com

GitHub Issues: Sorun bildirin



---

📄 Lisans

MIT License. Detaylar için **LICENSE** dosyasına bakın.


---

> 💙 Not: nexa-7.0-insomnia modeli empati ve insan anlama kapasitesine sahip, duygusal destek ve insan etkileşimi gerektiren uygulamalar için idealdir.

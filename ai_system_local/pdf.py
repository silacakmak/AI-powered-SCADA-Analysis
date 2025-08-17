

#!/usr/bin/env python3
"""
SCADA PDF Analiz Aracı (Yerel LLM - Ollama Gemma3:12B)
Bu script PDF'den SCADA verisini çıkarır ve yerel LLM ile yorumlar.
"""

import subprocess
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from datetime import datetime

# ======================
# GİRİŞ AYARLARI
# ======================

PDF_PATH = "data/2.pdf"   # PDF dosya yolunu buraya yaz
MODEL_NAME = "llama3.1:8b"        # Ollama'daki yerel model
OUTPUT_PATH = "output/llm_analysis.txt"

PROMPT_TR = """
Sen 20 yıllık SCADA sistem analizi deneyimine sahip bir koruma mühendisisin. PDF'deki verilerde **trip pickup seviyeleri veya kesici durumuyla ilgili doğrudan bilgi eksikse**, aşağıdaki adımları takip ederek **öncelikle en net ve önemli bulguları kısa ve madde madde listele**, sonra altına detaylı analiz ve açıklamaları yaz.

---

### Çıktı formatı:  

**Özet Bulgular:**  
- [Önemli nokta 1]  
- [Önemli nokta 2]  
- [Önemli nokta 3]  

---

**Detaylı Analiz:**  

#### 1. ÖN DEĞERLENDİRME (Zorunlu İlk Adım)  
- Tüm ölçülen parametreleri zaman serisi olarak özetle (voltaj, akım, frekans)  
- Ani değişimleri IEC 60255 standart hatası aralığı (±7.5%) içinde mi diye kontrol et  
- Kesici konum değişiklikleriyle eş zamanlı akım/voltaj değişimlerini eşleştir  

#### 2. TRIP PICKUP ANALİZİ (Dolaylı Yöntemler)  
- Aşırı akım bölgelerinde IEC 60255 ters zaman eğrilerine uyumunu kontrol et:  
  t(I) = TMS × [k / (I/Is)^α - 1]  
  (Is = ayar akımı, I = ölçülen akım)  
- Trip zamanı verisi yoksa, standart ters (NI) eğrisi varsayımıyla teorik hesaplama yap  
- Kesici açma anıyla röle alarmı arasındaki gecikme IEC 60255'de tanımlanan dinamik performans limitleri içinde mi kontrol et  

#### 3. OLASILIK TABANLI ARIZA TESPİTİ  
**YÜKSEK OLASILIK** (Veriler IEC 60255 ile %80+ uyumlu):  
- Örnek: "Ani 3I0 yükselmesi + voltaj çökmeleri → faz-toprak arızası (IEC 60255-3 göre çok ters zaman eğrisi tetiklenmiş olmalı)"  
- Kanıt: Ölçülen trip zamanı teorik hesapla ±7.5% hata sınırı içinde  

**ORTA OLASILIK** (Parçalı veri uyumu):  
- Örnek: "Harmonik distorsiyon + kısmi akım kesilmesi → yanlış röle tetiklemesi ihtimali (IEC 60255-149'da tanımlanan termal koruma sınırı aşılmış olabilir)"  
- Kanıt: Sadece voltaj dalgalanması var, kesici konum verisi eksik  

#### 4. VERİ EKSİKLİĞİ SENARYOLARI  
- Trip log'u yoksa: "IEC 60255 standart ayar eğrisi varsayımıyla analiz devam ediyor" notu ekle  
- Kesici konum bilgisi eksikse: "Ani akım sıfırlanması → kesici açmış kabul edilerek dinamik performans analizi yap"  
- Frekans verisi yoksa: "50Hz referans alınarak harmonik analiz yürütülüyor" uyarısı ver  

#### 5. SONUÇ  
- Arızanın türünü, koruma mekanizmalarının doğru çalışıp çalışmadığını ve sistemin genel durumunu değerlendir  
- Özet istatistikler ve grafik analizindeki verileri (SCADA Olay Özeti ve Röle Grafik Analizi) kullanarak raporu destekle  

---

### Acil Yapılacaklar (Kısa Vadeli)  
pdf üzerinden arızada elde edilen verilerle çok aşırı kısa bir şekilde acil eylem planı oluştur:
1. [Anında yapılması gereken eylem 1]  
2. [Anında yapılması gereken eylem 2]  
3. [Anında yapılması gereken eylem 3]  

---

### Uzun Vadeli Çözümler  
pdf üzerinden arızada elde edilen verilerle çok aşırı kısa bir şekilde sistemsel iyileştirme önerileri:
1. [Sistemsel çözüm 1]  
2. [Sistemsel çözüm 2]  
3. [Sistemsel çözüm 3]  

---


SCADA Veri Örneği:
\"\"\"
{scada_text}
\"\"\"
"""

# ======================
# METİN ÇIKAR
# ======================
genai.configure(api_key="AIzaSyD5c3r1Zh9KvhVU2UzJYyXiOiQpOVYGDj0")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    all_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"
    return all_text

# ======================
# OLLAMA YEREL LLM ÇAĞRISI
# ======================


# API anahtarını burada ayarla (örn: ortam değişkeninden veya doğrudan)


def query_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")  # veya istediğin model
    response = model.generate_content(prompt)
    return response.text

# ======================
# ANA FONKSİYON
# ======================
def main():
    print("📄 PDF verisi okunuyor...")
    text = extract_text_from_pdf(PDF_PATH)

    if not text.strip():
        print("❌ PDF içeriği boş veya okunamadı.")
        return

    print("🤖 LLM analiz başlatılıyor (gemini-2.5-flash)...")
    final_prompt = PROMPT_TR.format(scada_text=text[:4000])

    output = query_gemini(final_prompt)

    if output:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"✅ Analiz tamamlandı. Kaydedildi: {OUTPUT_PATH}")
    else:
        print("❌ LLM çıktısı alınamadı.")

if __name__ == "__main__":
    main()

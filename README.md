python -c "
readme_content = \"\"\"# Adli Görüntü Sahteciliği ve Deepfake Tespiti

**Öğrenci:** Samed Alp Arslan (220205012)  
**Ders:** Görüntü Adli Bilişimine Giriş

## 📄 Proje Özeti
Bu çalışmada adli bilişim kapsamında iki veri seti üzerinde **(i) kopyala-yapıştır sahteciliği (CoMoFoD)** ve **(ii) yüz manipülasyonu / deepfake (Celeb-DF)** tespiti için ikili sınıflandırma deneyleri gerçekleştirilmiştir.

Agresif olmayan bir ön-işleme katmanının (Gray-World + Median Filtre) ResNet-18 üzerindeki etkisi incelenmiştir.

## 📂 Veri Setleri
Projede kullanılan veri setleri (Boyut nedeniyle repoya dahil edilmemiştir, yerel `datasets/` klasöründe tutulmalıdır):
1. **CoMoFoD (Small v2):** Doğal görüntüler üzerinde copy-move varyasyonları.
2. **Celeb-DF:** Gerçek ve sahte videolardan alınmış kareler.

## 🛠️ Yöntem ve Mimari

### 1. Ön-İşleme (Preprocessing)
- **Gray-World:** Beyaz dengeleme ile kanal ortalamalarının eşitlenmesi.
- **Median Filtre (3x3):** İmpuls gürültünün bastırılması.

### 2. Sınıflandırma Modeli
- **Model:** ResNet-18 (ImageNet ön-eğitimli).
- **Konfigürasyon:** AdamW optimizatörü, Learning Rate: 1e-3, Epoch: 8, Batch: 32.

## 📊 Bulgular ve Sonuçlar

### Doğruluk (Accuracy) Tablosu

| Veri Seti | Senaryo | Doğruluk (Accuracy) |
|-----------|---------|---------------------|
| CoMoFoD | Sadece Model (None) | **%55.71** |
| CoMoFoD | GrayWorld + Median | %54.94 |

### Yorum
Ön-işleme adımı *forged* (sahte) görüntülerin tespitinde kısmi iyileşme sağlasa da, genel doğrulukta istatistiksel olarak anlamlı bir fark yaratmamıştır.

## 🚀 Kurulum ve Çalıştırma

Gerekli kütüphaneleri yüklemek için:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

Modeli eğitmek için:
\`\`\`bash
python scripts/train_classifier.py
\`\`\`
\"\"\"
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)
print('README.md basariyla olusturuldu.')
"

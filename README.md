# Kedi & Köpek Sınıflandırma Projesi

VGG16 transfer öğrenme modeli kullanılarak kedi ve köpek görüntülerinin sınıflandırılması projesi.


## 📂 Proje Yapısı
project/
├── data/
│ ├── images/ # Tüm eğitim görüntüleri (.png)
│ └── annotations/ # XML etiket dosyaları
├── model.keras # Eğitilmiş model dosyası
└── camera.py # kamerayı açıp test etme



## 🛠 Ön Gereksinimler
- Python 3.8+
- Gerekli kütüphaneler:
  ```bash
  pip install tensorflow numpy pandas pillow scikit-learn matplotlib seaborn
Veri seti yapısı:

Her görüntü için .png dosyası

Her görüntü için aynı isimde XML etiket dosyası (Pascal VOC formatı)

🧠 Temel Özellikler
Transfer Öğrenme ile VGG16 modeli

Veri Artırma (Data Augmentation)

Erken Durdurma (Early Stopping)

Sınıflandırma Raporu ve Karışıklık Matrisi

⚙️ Model Mimarisi
python
Copy
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 vgg16 (Functional)          (None, 7, 7, 512)         14714688  
                                                                 
 global_average_pooling2d (  (None, 512)               0         
 GlobalAveragePooling2D)                                         
                                                                 
 dense (Dense)               (None, 256)               131328    
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 1)                 129       
                                                                 
=================================================================
Total params: 14,879,041
Trainable params: 164,353
Non-trainable params: 14,714,688
🚀 Kullanım
Veri setini hazırla:

Görüntüleri data/images/ klasörüne yerleştir

XML etiket dosyalarını data/annotations/ klasörüne yerleştir

Modeli eğit:

python
Copy
python cat_dog_classifier.py
Eğitim sonuçları:

Model kaydı: model.keras

Performans metrikleri konsol çıktısında

📊 Örnek Sonuçlar
Copy
Test accuracy: 0.9275

Sınıflandırma Raporu:
              precision    recall  f1-score   support

           0       0.93      0.92      0.93       203
           1       0.92      0.93      0.93       197

    accuracy                           0.93       400
   macro avg       0.93      0.93      0.93       400
weighted avg       0.93      0.93      0.93       400
Confusion Matrix

📌 Önemli Notlar

XML etiketlerinde sadece 'cat' ve 'dog' etiketleri desteklenir.

.keras uzantılı dosyayı .tflite ye çevirmek için convert.py dosyasını kullanabilirsiniz.

Görüntüler otomatik olarak 224x224 boyutuna yeniden ölçeklendirilir

🌟 Geliştirme İçin Öneriler
Daha büyük veri seti kullanımı

Farklı pretrained modellerin denenmesi (ResNet, EfficientNet)

Hiperparametre optimizasyonu

TensorBoard entegrasyonu

Dockerizasyon

🌟 Katkıda bulunmaktan çekinmeyin! PR'larınızı bekliyorum.

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Modeli yükle
model = load_model('model.keras')

# Görüntü verisi oluşturucu
test_datagen = ImageDataGenerator(rescale=1./255)

# MODELİN SON KATMANINI KONTROL ET
if model.output_shape[-1] == 1:
    class_mode = 'binary'
else:
    class_mode = 'categorical'

# Test veri setini yükle
test_generator = test_datagen.flow_from_directory(
    r'C:\data\test_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode=class_mode  # Modelin çıkışına uygun şekilde ayarlanıyor
)

# Tahminleri al
y_pred = model.predict(test_generator)

# Eğer modelin çıktısı (None, 1) ama veri (None, 2) formatında ise dönüştür
if y_pred.shape[1] == 1 and class_mode == 'categorical':
    y_pred = np.hstack([1 - y_pred, y_pred])  # (None, 1) → (None, 2)

# Modeli değerlendir
score = model.evaluate(test_generator)

print(f"Loss: {score[0]}")
print(f"Accuracy: {score[1]}")

import tensorflow as tf

# 1️⃣ Keras modelini yükle
model = tf.keras.models.load_model("model.keras")

# 2️⃣ TFLite Converter kullanarak modeli dönüştür
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3️⃣ Çıktıyı kaydet
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Dönüştürme tamamlandı: model.tflite dosyası oluşturuldu.")

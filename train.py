import os

import tensorflow as tf


import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def parse_xml(xml_file):
    """ XML dosyasını okuyup etiketleri çıkarır. """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels = []
    for member in root.findall('object'):
        labels.append(member.find('name').text)
    return labels


def load_dataset(image_dir, annotation_dir):
    """ Görüntüleri yükler ve ilgili etiketleri belirler. """
    images = []
    labels = []

    for image_file in os.listdir(image_dir):
        if image_file.endswith('.png'):
            
            base_name = os.path.splitext(image_file)[0]  
            annotation_path = os.path.join(annotation_dir, f"{base_name}.xml")

            if not os.path.exists(annotation_path):
                print(f"Uyarı: {annotation_path} bulunamadı, atlanıyor.")
                continue

            labels_list = parse_xml(annotation_path)
            if 'cat' in labels_list:
                label = 0  
            elif 'dog' in labels_list:
                label = 1 
            else:
                print(f"Uyarı: {image_file} için geçerli bir etiket bulunamadı, atlanıyor.")
                continue

           
            image = Image.open(os.path.join(image_dir, image_file))
            image = image.resize((224, 224))
            image = image.convert('RGB') 
            image = np.array(image) / 255.0  

           
            images.append(image)
            labels.append(label)

    if not images or not labels:
        raise ValueError("Hata: Veri seti boş! Lütfen image ve annotation dosyalarını kontrol edin.")

    return np.array(images), np.array(labels)


image_dir = r'C:\data\veriler\images'
annotation_dir = r'C:\data\veriler\annotations'


images, labels = load_dataset(image_dir, annotation_dir)
print(f"Yüklenen görüntü sayısı: {len(images)}, Yüklenen etiket sayısı: {len(labels)}")
print(f"Görüntülerin boyutu: {images.shape}")
print(f"Etiketlerin boyutu: {labels.shape}")


X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


datagen.fit(X_train)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)


history = model.fit(datagen.flow(X_train, y_train,batch_size = 32),
                    epochs=60,  
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])


test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int) 

print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

model.save('model.keras')
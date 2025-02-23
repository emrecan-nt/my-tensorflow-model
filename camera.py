import cv2
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("model.keras")


cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    image = cv2.resize(frame, (224, 224))
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  

    
    prediction = model.predict(image)[0][0]
    confidence = round(max(prediction, 1 - prediction) * 100, 2)  
    label = "Kedi" if prediction < 0.5 else "Kopek"

   
    threshold = 99.5
    if confidence < threshold:
        label = "Nesne Yok"

    
    text = f"{label} ({confidence:.2f}%)"
    cv2.putText(frame, text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Prediction", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

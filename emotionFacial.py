import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# ===========================
# 1️⃣ Tạo dataset TensorFlow
# ===========================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Chuẩn hóa [0,1]
train_ds = train_ds.map(lambda x, y: (x/255.0, y))
val_ds = val_ds.map(lambda x, y: (x/255.0, y))

# ===========================
# 2️⃣ Xây dựng CNN Model
# ===========================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(7, activation='softmax')
])

# ===========================
# 3️⃣ Compile + Train
# ===========================
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

# ===========================
# 4️⃣ Lưu và test thử
# ===========================
model.save("facial_emotion_model.h5")
print("\n✅ Model đã được train xong và lưu thành công!")

# Dự đoán thử 1 ảnh
# img_path = "dogCat.png"
# img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
# img_array = tf.keras.utils.img_to_array(img) / 255.0
# img_array = np.expand_dims(img_array, axis=0)

# prediction = model.predict(img_array)
# print(prediction)

# class_names = train_ds.class_names
# predicted_index = np.argmax(prediction)
# predicted_class = class_names[predicted_index]

# print(f"Ảnh này là: {predicted_class} 🐦🐱🐶")
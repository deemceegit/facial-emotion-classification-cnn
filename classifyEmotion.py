from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

model = load_model("facial_emotion_model_02.h5")
print("model loaded sucessfully!!!")

# print structure
print("start print structure")
model.summary()
print("done print structure")

from tensorflow.keras.utils import load_img, img_to_array

IMG_SIZE = (128, 128)

img_path = "randomImgTest/test_surprise.png"

img = load_img(img_path, target_size=IMG_SIZE)  # same szie w model
img_array = img_to_array(img) / 255.0           # normalize [0,1]
img_array = np.expand_dims(img_array, axis=0)   # add batch dimension (1,128,128,3)

prediction = model.predict(img_array)   # output looks like: [[0.01531001 0.00125472 0.9834352 0.123123 0.123123 0.123 0.1293]]
print(prediction)

class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
predicted_index = np.argmax(prediction)         # choose the highest percentage
predicted_class = class_names[predicted_index]

print(f"the emotion is: {predicted_class}")
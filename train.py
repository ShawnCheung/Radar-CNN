import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from PIL import Image
import numpy as np
import pdb, os


images = os.listdir("./train-c")[0:1000]
x_train = []
y_train = []
for i in images:
    x_train.append(np.array(Image.open("./train-k/"+i).convert('RGB')))
    y_train.append(np.array(Image.open("./train-c/"+i).convert('RGB')))
x_train = np.array(x_train)
y_train = np.array(y_train)
pdb.set_trace()

model = keras.Sequential()
model.add(keras.layers.Conv2D(16, (3, 3), padding="same", activation='relu', input_shape=(512, 512, 3)))
model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(keras.layers.Conv2D(16, (3, 3), padding="same", activation='relu'))
model.add(keras.layers.Conv2D(3, (3, 3), padding="same", activation='relu'))

model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt)

model.fit(x_train, y_train, batch_size=32, epochs=10)
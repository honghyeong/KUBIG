# import the necessary packages
import argparse
import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt


# construct the argument parse and parse the arguments
# 실행 할때마다 변경이 있을 때, 실행시킬때마다 설정을 바꿔주는 부분.
ap=argparse.ArgumentParser(description='Build a smile detection model')
ap.add_argument('-d','--dataset',required=True,help='path to input dataset')
ap.add_argument('-m','--model',required=True,help='path to output model')
args=vars(ap.parse_args())

# get data directories

# base_dir='datasets'
base_dir=args['dataset']

train_dir=os.path.join(base_dir,'train_folder')
test_dir=os.path.join(base_dir,'test_folder')

# Image Data Generator
train_datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rescale=1./255

)

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(64,64),
    batch_size=20,
    class_mode='binary'
)

validation_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=(64,64),
    batch_size=20,
    class_mode='binary'
)

# build keras model
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

# compile model
print('[INFO] compiling model...')
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])

# fit model and save
print('[INFO] training network...')
history=model.fit_generator(train_generator,
                            steps_per_epoch=100,
                            epochs=15,
                            validation_data=validation_generator,
                            validation_steps=50)

print("[INFO] serializing network ...")
model.save(args["model"])

# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), history.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 15), history.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, 15), history.history["acc"], label = "acc")
plt.plot(np.arange(0, 15), history.history["val_acc"], label = "val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
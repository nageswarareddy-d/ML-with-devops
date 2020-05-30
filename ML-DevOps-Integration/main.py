#!/usr/bin/python
from layers import *
cps()
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        '/tmp/cnn_dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        '/tmp/cnn_dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=8000,
        epochs=4,
        validation_data=test_set,
        validation_steps=800)

accuracy= model.history.history['accuracy']
accuracy=str(accuracy)[1:-1]
print(accuracy)
import sys
sys.stdout = open('/tmp/accuracy.txt','wt')
print(accuracy)


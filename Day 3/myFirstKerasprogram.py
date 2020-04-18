'''
This is my first Keras program. Doing and learning with the help of Deep Learning with Python book by Francis Chillote
'''

from keras.datasets import mnist
from keras import layers,models
import matplotlib.pyplot as plt
(train_images, train_labels),(test_images,test_labels)= mnist.load_data()

#print("Training Set has Dimensions of",train_images.shape)
#print("Testing Set has Dimensions of",test_images.shape)

'''
Workflow is that we feed Neural Networks the training data,training images,training labels. The network will learn to associate images and labels. Then we will ask network to produce predictions for test_images and we will verify them from test_labels
'''


network = models.Sequential()

network.add(layers.Dense(512,activation='tanh',input_shape = (28*28,)))
network.add(layers.Dense(10,activation='softmax'))
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])   
train_images = train_images.reshape((60000,28*28))

train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical
#to categorically encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
network.fit(train_images,train_labels,epochs=5,batch_size=124)

test_loss,test_acc = network.evaluate(test_images,test_labels)

print("test_acc:",test_acc)


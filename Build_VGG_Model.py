#!/usr/bin/env python
# coding: utf-8

# In "vgg_transfer_learning" example, we see how to do transfer learning using vgg16 model.
# In this code, we are going to build the vgg model from scratch and use Keras ImageDataGenerator Class for preprocessing.

# In[ ]:


from keras.models import Model
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
import keras


# In[ ]:


from keras.datasets import cifar10 

# import dataset cifar10, which contains "50,000 32x32 color training images, labeled over 10 categories, 
# and 10,000 test images".
# The loaded data are already in the form of numpy array
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_classes = 10 # 10 categories in total

# y values are integers. We need to convert them to one-hot vectors for 
# training since the last layer of our model is "softmax".
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[ ]:


# define vgg block
# layer_in: the output layer from last block
# n_filters: number of filters you want to use
# n_conv: how many conv layers you want to apply
def define_vgg_block(layer_in, n_filters, n_conv):
    for _ in range (n_conv):
        layer_in = Conv2D(n_filters, (3,3), padding = 'same', activation = 'relu')(layer_in)
    layer_out = MaxPooling2D((2,2), strides = (2,2))(layer_in)
    return layer_out


# In[ ]:


# Build the vgg model
# The input shape of cifar10 is rgb (32,32)
input_tensor = Input(shape = (32, 32, 3))

# The reason we only use 2 blocks is that:
# the input image size is small(32,32), when we apply twice maxpooling, it becomes (8,8)
# This (8,8), to some extend, indicates it extract 8x8 = 64 features vectors from the previous image. 
# And it's not bad using the 64 features to make 10 categories.
# However, if we apply two more maxpoolings, only 2x2=4 features left, which may definitely not be a good choice.
block_1 = define_vgg_block(input_tensor, 64, 2)
block_2 = define_vgg_block(block_1, 128, 2)

flat_layer_1 = Flatten()(block_2)

# Here we add a 128 neurons layer before the 10 softmax. Becasue:
# The output of previous layer is 8192 neurons, which contains too tiny and subtle info for the 10 classes
# we add the 128 neuron layer to combine and filter the info, and then feed them to the 10 classes layer.
# More accuracy for prediction.
flat_layer_2 = Dense(128, activation = 'relu')(flat_layer_1)
flat_layer_3 = Dense(10, activation = 'softmax')(flat_layer_2)
model = Model(input = input_tensor, output = flat_layer_3)
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()


# In[ ]:


# We use ImageDataGenerator to "generate batches of tensor image data with real-time data augmentation". 
from keras.preprocessing.image import ImageDataGenerator

# create instance
train_dataGen = ImageDataGenerator(rescale = 1.0/255.0, width_shift_range = 0.1, height_shift_range = 0.1, 
                             horizontal_flip = True)
test_dataGen = ImageDataGenerator(rescale = 1.0/255.0)

# Train the model
history = model.fit_generator(train_dataGen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=20, 
                    validation_data=test_dataGen.flow(x_test, y_test), validation_steps=len(x_test)/32, verbose = 1)


# In[ ]:


model.save('img_clf_cifar10')
model.save_weights('img_clf_cifar10_weights.h5')


# In[ ]:





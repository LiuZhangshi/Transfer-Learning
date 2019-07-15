#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras import applications
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras import optimizers 
import keras.datasets
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt


# In[2]:


# Specify input shape. Since we are using dataset cifar10, the input size should be (32,32,3)
input_tensor = Input(shape = (32, 32, 3))

# Borrow vgg model as the base model. Use partial pre-trained weights, delete the top layer
vgg_model = applications.VGG16(weights = 'imagenet', 
                               include_top = False, 
                               input_tensor = input_tensor)


# In[3]:


# show the layers in layer_dict
layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

# If you don't want to use the complete vgg model, you can pick one layer as your last vgg block
# Then add your own fc layers and softmax
# Here I picked block3_pool. It's probably not the best choice for the application. I just wanna show how to do this.
out_layer = layer_dict['block3_pool'].output

# Stack new layers 
# First flatten the conv layer, then add fc layer, then softmax. Again, it's may not be a good choice, just an example.
out_layer = Flatten()(out_layer)

# Here I didn't connect the flatten layer directly to the 10 softmax. Instead I added one 'buffer' layer to improve accuracy.
# This 'buffer' layer is for managing and filtering info from flatten, which contains 4096 neurons.
out_layer = Dense(128, 
          kernel_initializer='random_uniform',
          bias_initializer='zeros', 
          activation = 'relu')(out_layer)
out_layer = Dense(10, 
          kernel_initializer='random_uniform',
          bias_initializer='zeros',
          activation = 'softmax')(out_layer)


# In[4]:


cur_model = Model(input = vgg_model.input, output = out_layer)


# In[5]:


# Decide whether you want to change the pre-trained weights or not.
# Here I keep the first 10 layers' weights
for layer in cur_model.layers[:10]:
    layer.trainable = False
cur_model.summary()


# In[6]:


# specify optimizor, here Adam with learning rate 0.001
optim = keras.optimizers.Adam(lr=0.001)

# model compile
# The loss func, binary_crossentropy for 2 labels, 
# and categorical_crossentropy for more than 2 labels. Here we have 10 categories
cur_model.compile(loss = 'categorical_crossentropy',
                 optimizer = optim,
                 metrics = ['accuracy'])


# In[7]:


from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[8]:


num_classes = 10

# The y values are integers, but our model softmax output is probability distribution. Each component
# will be interval (0,1) and sum up to 1.
# Here we use to_categorical to convert y to one-hot vectors before training
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[9]:


# The loaded x data are integers, we need to convert them to float32 type for the following preprocessing actions.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# data scaling 
x_train /= 255.0
x_test /= 255.0

# The vgg author wrote in the paper that:
# "The only preprocessing we do is subtracting the mean RGB value, computed on the training set, from each pixel."
# Thus the 'preprocess_input' func imported from 'keras.applications.vgg' will only do the mean subtract for preprocess
#x_train = preprocess_input(x_train)
#x_test = preprocess_input(x_test)


# In[ ]:


#from keras.callbacks import LambdaCallback
#print_weights = LambdaCallback(on_epoch_begin=lambda batch, logs: print(cur_model.layers[9].get_weights()))


# In[10]:


batch_size = 32
epochs = 15
cur_model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 1)


# In[11]:


cur_model.evaluate(x_test, y_test)


# In[37]:


# prediction-----------------------------------------------------------------
# 'load_img' returns a PIL image instance.
image = load_img('truck.jpg', target_size=(32, 32), color_mode = 'rgb')
plt.imshow(image)

# Keras provides the img_to_array() function for converting a loaded image in PIL format 
# into a NumPy array for use with models.
image = img_to_array(image)

# The input of the model should be (num_samples, img_shape[0], img_shape[1], img_shape[2])
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# Convert integers to float32
image = image.astype('float32')

# preprocessing
image /= 255.
#image = preprocess_input(image)

pred = cur_model.predict(image)


# In[32]:


import numpy as np
_, i = np.unravel_index(pred.argmax(), pred.shape)

if i == 0:
    print("It's a airplane!")
elif i == 1:
    print("It's a automobile!")
elif i == 2:
    print("It's a bird!")
elif i == 3:
    print("It's a cat!")
elif i == 4:
    print("It's a deer!")
elif i == 5:
    print("It's a dog!")
elif i == 6:
    print("It's a frog!")
elif i == 7:
    print("It's a horse!")
elif i == 8:
    print("It's a ship!")
elif i == 9:
    print("It's a truck!")


# In[34]:


cur_model.save_weights('vgg_cifar10.h5')


# In[ ]:





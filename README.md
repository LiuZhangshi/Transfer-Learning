# Transfer-Learning
## VGG transfer learning
This code use pretrained model VGG16 as the basis to buid a new image recognition model. 
In this example, we did the "transfer learning" by borrowing the first 3 convolutional blocks of the model, and then adding 2 fully connected layers.
We also borrowed the weights of the 2nd and 3rd Convnet layers. And trained other parameters using CIFAR10.
CIFAR10 has a small image classification Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images, which can be accessed directly from Keras. 

## Build vgg block
In this code, we built the vgg model from scratch and use Keras ImageDataGenerator Class for preprocessing.

References:

1. Very Deep Convolutional Networks for Large-Scale Image Recognition. 2014
2. https://keras.io/examples/cifar10_cnn/
3. https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/

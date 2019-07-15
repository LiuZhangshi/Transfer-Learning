# Transfer-Learning
## VGG transfer learning

VGG16 model is the winner of ILSVRC 2014 competition for 1000 categories image classification. It consists of 16 convolutional layers. It has very uniform architecture of "vgg block" which includes 2 conv layers followed by one pooling layer.

This code uses pretrained model VGG16 as the basis to buid a new image classification model trained on dataset cifar10, which contains smaller size images (32,32) and only 10 categories. 

In this example, we did the "transfer learning" by borrowing the first 3 convolutional blocks of the model, and then adding 2 fully connected layers.

We also borrowed the weights of the first 10 layers. And trained other parameters in fully connected layers.

CIFAR10 has a small image classification Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images, which can be accessed directly from Keras. 

## Build vgg block
In this code, we built the vgg model from scratch and use Keras ImageDataGenerator Class for preprocessing.

References:

1. Very Deep Convolutional Networks for Large-Scale Image Recognition. 2014
2. https://keras.io/examples/cifar10_cnn/
3. https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/

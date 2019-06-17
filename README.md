# Transfer-Learning
This code using pretrained model VGG16 as the basis to buid a new image recognition model. 
VGG16 model trained on over 1 million images to classify 1000 categories.
In this example, we did the "transfer learning" by borrowing the model construction of the first 3 blocks of Convnet layers, 
and then added 2 fully connected layers.
We also borrowed the weights of the 2nd and 3rd Convnet layers. And trained other parameters using CIFAR10.
CIFAR10 has a small image classification Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images, 
which can be accessed directly from Keras. 

References:

1. Very Deep Convolutional Networks for Large-Scale Image Recognition. 2014
2. https://keras.io/examples/cifar10_cnn/
3. https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/

# Load pickled data
import pickle

# Done: fill this in based on where you saved the training and testing data
training_file = "dataset/train.p"
testing_file = "dataset/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

print("Pickle files loaded succesfully")

### To start off let's do a basic data summary.

# Done: number of training examples
n_train = len(X_train)

# Done: number of testing examples
n_test = len(X_test)

# Done: what's the shape of an image?
image_shape = train['sizes'][0]

# Done: how many classes are in the dataset
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Preprocess the data here.
### Feel free to use as many code cells as needed.
import numpy as np
import cv2

def preprocess(image):
    # presets
    img_size = 48 # image height/width
    border = img_size // 10 # 10% borders
    new_size = img_size + (2 * border)
    new_shape = (new_size, new_size) # shape of the target image
    
    # 1. Resample image
    img_rs = np.zeros(new_shape)
    # a. If image size is new_size, do nothing:
    if image.shape == new_shape:
        img_rs = image
    # b. If image is larger, shrink using recommended downsizing interpolation
    elif image.shape > new_shape:
        img_rs = cv2.resize(src = image, dsize = new_shape, interpolation = INTER_AREA)
    # c. If image is smaller, zoom using bilinear interpolation, which is the default
    else: 
        img_rs = cv2.resize(src = image, dsize = new_shape)
    
    # 2. Crop the borders of the image
    img_48 = img_rs[border:-border,border:-border]
    
    # 3. Normalize image as per sermanet'11
    # a. Convert image to YUV space
    yuv = cv2.cvtColor(img_48, cv2.COLOR_BGR2YUV)
    # b. Normalize Y channel using adaptive histogram equalization, chosen from Ciresan'12
    clahe = cv2.createCLAHE(clipLimit = 5, tileGridSize=(6,6))
    yuv[:,:,0] = clahe.apply(yuv[:,:,0])
    norm = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    return norm

def rescale(matrix):
    # Maps an RGB image matrix from 0:255 to -1:1
    nmatrix = np.zeros_like(matrix, dtype=np.float32)
    if len(matrix.shape) == 2:
    # Greyscale:
        return (2. * matrix / 255.) - 1.
    # Colour:
    for i in range(matrix.shape[-1]):
        nmatrix[:,:,i] = (2. * matrix[:,:,i] / 255.) - 1.
    return nmatrix
    
# Preparing the data
X_train_prep = np.array([rescale(preprocess(img)) for img in X_train])
X_test_prep = np.array([rescale(preprocess(img)) for img in X_test])

# Repeating the basic data summary for the preprocessed data.

n_train = len(X_train_prep)
n_test = len(X_test_prep)
image_shape = X_train_prep[0].shape
    
### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf

### Parameters
img_size = 48

# convolutions
stc   = [1, 1, 1, 1]     # stride used in convolution steps
conv1 = (7, 7, 3, 100)   # shape of the first convolution filter
conv2 = (4, 4, 100, 150) # shape of the second convolution filter
conv3 = (4, 4, 150, 250) # shape of the third convolution filter

# max pooling
stp  = [1, 2, 2, 1] # stride used in max pooling steps
pwnd = (1, 2, 2, 1) # shape of the max pooling windows

# fully connected layers
mult1 = (2250, 300) # fully connected layer weights shape
mult2 = (300, 43)   # output layer weights shape


### Placeholders
inputs = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 3))
labels = tf.placeholder(tf.float32, shape=(None, n_classes))

### Filters/Weights:
weights = {
    "conv1": tf.Variable(tf.random_normal(shape = conv1)), # convolution filter
    "conv2": tf.Variable(tf.random_normal(shape = conv2)), # convolution filter
    "conv3": tf.Variable(tf.random_normal(shape = conv3)), # convolution filter
    "mult1": tf.Variable(tf.random_normal(shape = mult1)), # fully connected layer weights
    "mult2": tf.Variable(tf.random_normal(shape = mult2))  # output layer weights
          }
### Biases
biases = {
    "conv1": tf.Variable(tf.random_normal(shape = conv1[-1:])),
    "conv2": tf.Variable(tf.random_normal(shape = conv2[-1:])),
    "conv3": tf.Variable(tf.random_normal(shape = conv3[-1:])),
    "mult1": tf.Variable(tf.random_normal(shape = mult1[-1:])),
    "mult2": tf.Variable(tf.random_normal(shape = mult2[-1:]))
         }

### Operations:
# first convolution
layer1 = tf.nn.conv2d(input = inputs, filter = weights["conv1"],
                      strides = stc, padding = "VALID")
layer1 = tf.nn.bias_add(layer1, biases["conv1"])

# first max pooling
layer2 = tf.nn.max_pool(value = layer1, ksize = pwnd,
                        strides = stp, padding = "VALID")
layer2 = tf.tanh(layer2)

# second convolution
layer3 = tf.nn.conv2d(input = layer2, filter = weights["conv2"],
                      strides = stc, padding = "VALID")
layer3 = tf.nn.bias_add(layer3, biases["conv2"])

# second max pooling
layer4 = tf.nn.max_pool(value = layer3, ksize = pwnd,
                        strides = stp, padding = "VALID")
layer4 = tf.tanh(layer4)

# third convolution
layer5 = tf.nn.conv2d(input = layer4, filter = weights["conv3"],
                      strides = stc, padding = "VALID")
layer5 = tf.nn.bias_add(layer5, biases["conv3"])

# third max pooling
layer6 = tf.nn.max_pool(value = layer5, ksize = pwnd,
                        strides = stp, padding = "VALID")
layer5 = tf.tanh(layer5)

# fully connected layer
layer7 = tf.matmul(tf.reshape(layer6, [-1, 2250]), weights["mult1"])
layer7 = tf.tanh(tf.nn.bias_add(layer7, biases["mult1"]))

# output layer
output = tf.matmul(layer7, weights["mult2"])
output = tf.nn.softmax(tf.nn.bias_add(output, biases["mult2"]))

print("Architecture built successfully!")

### Train your model here.
### Feel free to use as many code cells as needed.


### TrainingSetup

# Parameters:
learning_rate = 0.5
training_epochs = 1
batch_size = 500

# Optimization
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)

### Training
# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    # Training cycle
    for _ in range(training_epochs):
        for i in range(0, n_train, batch_size):
            batch_x = X_train_prep[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            print(type(batch_x))
            print(type(batch_y))
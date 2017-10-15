# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/center_2017_10_14_14_53_29_045.jpg "Recovery Image (1)"
[image4]: ./examples/center_2017_10_14_14_54_39_331.jpg "Recovery Image (2)"
[image5]: ./examples/center_2017_10_14_14_55_33_701.jpg "Recovery Image (3)"
[image6]: ./examples/center_1_0,0,0.0.jpg "Normal Image"
[image7]: ./examples/center_1_-24,0,-0.192.jpg "Shifted Image"
[image8]: ./examples/center_2017_10_14_14_56_50_899.jpg "Center Driving (1)"
[image9]: ./examples/center_2017_10_14_14_57_34_602.jpg "Center Driving (2)"
[image10]: ./examples/center_2017_10_14_14_57_54_295.jpg "Center Driving (3)"
[image11]: ./examples/center_2017_10_14_14_56_39_426.jpg "Center Driving (4)"

[chart0]: ./examples/network_diagram.png "Network Diagram"
[chart1]: ./examples/learning_1_.png "Learning Loss (1)"
[chart2]: ./examples/learning_2_.png "Learning Loss (2)"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

#### Required Files

* See files below.

#### Quality of Code

* **Is the code functional?** -- Yes.
* **Is the code usable and readable?** -- It's commented in order to enhance code readibility. The generator is in place.

#### Model Architecture and Training Strategy

* **Has an appropriate model architecture been employed for the task?** -- A multilayer convolutional and squential model is in use along with dropout and activation function.
* **Has an attempt been made to reduce overfitting of the model?** -- See dropout and low learing rate.
* **Have the model parameters been tuned appropriately?** -- The model delivers the output desired.
* **Is the training data chosen appropriately?** -- In addition to the training data provided several other scenarios were covered.

#### Architecture and Training Documentation

* **Is the solution design documented?** -- See this file.
* **Is the model architecture documented?** -- See chart below.
* **Is the creation of the training dataset and training process documented?** -- See below.

#### Simulation

* **Is the car able to navigate correctly on test data?** -- The car is not leavin the track surface, as seen in the attached [video](https://github.com/cscsatho/CarND-Behavioral-Cloning-P3/blob/master/run10.mp4).

### Files Submitted & Code Quality

#### 1. Submitted files

My project includes the following files:

* [model.py](https://github.com/cscsatho/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to create and train the model
* [drive.py](https://github.com/cscsatho/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/cscsatho/CarND-Behavioral-Cloning-P3/blob/master/model.h5) containing a trained convolution neural network
* [run10.mp4](https://github.com/cscsatho/CarND-Behavioral-Cloning-P3/blob/master/run10.mp4) is a test video (10 mph)
* [run15.mp4](https://github.com/cscsatho/CarND-Behavioral-Cloning-P3/blob/master/run15.mp4) is a test video (15 mph)

#### 2. Submission includes functional code

Model generation
```sh
python model.py
```

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Network architecture:

![alt text][chart0]

The model includes exponential linear unit layers to introduce nonlinearity. (It has been shown that ELUs can obtain higher classification accuracy than ReLUs.)
The data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.
The model was trained and validated on different data sets to ensure that the model was not overfitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, with learning rate of 0.0001.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I used a combination of center lane driving, recovering from the left and right sides of the road.
Additionally I was recording a backwards driving on the track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I followed Nvidia's self diving car CNN, and finetuned it for this specific task.
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it includes dropout.

Then I further augmented the training data (flipping, shifting images, using left/right camera images, etc.)

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I recorded data for the uncertain parts of the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road at 10 mph. When I'm increasing the speed to 15 mph the vehice starts to swing a little bit, but still it stays on track.

#### 2. Final Model Architecture

The final model architecture was alerady shown above.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help to generalize.

I also shifted random images with random number of pixels along the x axis and adjusted the steering ange accordingly:

![alt text][image6]
![alt text][image7]

After the collection process, I had 40,542 data points. After preprocessing and augmentation it became 129,732.
I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][chart1]
![alt text][chart2]

### 4. Training printout

```
(carnd-term1) carnd@ip-172-31-38-68:~/CarND-Behavioral-Cloning-P3$ python model.py
Using TensorFlow backend.
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so locally
Parsing file data/driving_log_stock.csv with 8037 lines...
Parsing file data/driving_log_fwd.csv with 3031 lines...
Parsing file data/driving_log_bwd.csv with 1405 lines...
Parsing file data/driving_log_drift2.csv with 676 lines...
Parsing file data/driving_log_curve.csv with 366 lines...
c_1: (None, 66, 200, 3) (None, 31, 98, 24)
c_2: (None, 66, 200, 3) (None, 14, 47, 36)
c_3: (None, 66, 200, 3) (None, 5, 22, 48)
c_4: (None, 66, 200, 3) (None, 3, 20, 64)
c_5: (None, 66, 200, 3) (None, 1, 18, 64)
flt: (None, 66, 200, 3) (None, 1152)
d_1: (None, 66, 200, 3) (None, 1152)
d_2: (None, 66, 200, 3) (None, 100)
d_3: (None, 66, 200, 3) (None, 50)
d_4: (None, 66, 200, 3) (None, 10)
d_5: (None, 66, 200, 3) (None, 1)
Epoch 1/4
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties:
name: GRID K520
major: 3 minor: 0 memoryClockRate (GHz) 0.797
pciBusID 0000:00:03.0
Total memory: 3.94GiB
Free memory: 3.91GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GRID K520, pci bus id: 0000:00:03.0)
129732/129732 [==============================] - 111s - loss: 0.0361 - val_loss: 0.0268
Epoch 2/4
129732/129732 [==============================] - 109s - loss: 0.0276 - val_loss: 0.0237
Epoch 3/4
129732/129732 [==============================] - 109s - loss: 0.0255 - val_loss: 0.0239
Epoch 4/4
129732/129732 [==============================] - 109s - loss: 0.0242 - val_loss: 0.0207
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1152)          0           flatten_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1152)          1328256     dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 1152)          0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           115300      dropout_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]
====================================================================================================
Total params: 1,580,475
Trainable params: 1,580,475
Non-trainable params: 0
____________________________________________________________________________________________________
None
```

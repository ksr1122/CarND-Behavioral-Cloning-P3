# **Behavioral Cloning** 


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Model Visualization"
[image2]: ./examples/center_lane_driving.jpg "Center Lane Driving"
[image3]: ./examples/left_lane_driving.jpg "Recovery Image"
[image4]: ./examples/right_lane_driving.jpg "Recovery Image"
[image5]: ./examples/normal.jpg "Normal Image"
[image6]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used NVIDIA's pipeline which was discussed in the previous lesson which basically consists of normalization followed by five convolution layers and then followed by four fully connected layers. Also there is relu activation at the end of the every convolution layers.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on the default provided data sets including the images from all the cameras (original & flipped) along with some additional recovery images captured manually to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (**model.compile(loss='mse', `optimizer='adam'`)**).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model by autonomous vehicle team in NVIDIA's. I thought this model might be appropriate because it is already tested and proven model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set and also on the validation set. This implied that the training data was not sufficient for recovery laps. 

To combat that, I included also the left and right camera images into training. Then I used the flipped version of all those 3 images as well.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 76-96) consisted of a convolution neural network with the following layers and layer sizes

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 160x320x3 RGB image   					    |
| Cropping2D       		| output 65x160x3   					        |
| Lambda           		| output 65x160x3   					        |
| Convolution 5x5     	| 2x2 stride, valid padding                     |
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding                     |
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding                     |
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding                     |
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding                     |
| RELU					|												|
| Flatten       		|   									        |
| Fully connected		| output 100 									|
| Fully connected		| output 50  									|
| Fully connected		| output 10  									|
| Fully connected		| output 1   									|

#### 3. Training Set & Training Process

To capture good driving behavior, I first took two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then took the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to avoid and recover from running out of lane. These images show what a recovery looks like starting from left and right:

![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would enhance the training dataset. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

From the collection dataset, I had 24108 number of data points. I then preprocessed this data by cropping the image 
70px on top and 25px on bottom using the keras `Cropping2D`. Followed by normalization with keras `Lambda` function with `lambda x: x / 127.5 - 1.0` as argument.

I used scikit utility to randomly shuffle the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the comparitively better minimum loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.

My training history is visualized as shown below.

![alt text][image1]

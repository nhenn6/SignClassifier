# **Traffic Sign Recognition** 

## Writeup by Noah Henning

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/rotation.png "Rotation"
[image4]: ./TestImages/arrowforward.jpg "Traffic Sign 1"
[image5]: ./TestImages/Do-Not-Enter.jpg "Traffic Sign 2"
[image6]: ./TestImages/speed30.jpg "Traffic Sign 3"
[image7]: ./TestImages/stop.jpg "Traffic Sign 4"
[image8]: ./TestImages/Yield.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799 images.
* The size of the validation set is 4,410 images.
* The size of test set is 12,630 images.
* The shape of a traffic sign image is 32x32x3.
* The number of unique classes/labels in the data set is 42.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data. The number of images of each type of sign is quite unbalanced!

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because colors may fade over time, but the shapes of signs are generally constant. It might reduce processing time as well, but I don't know that for sure. 

Here is an example of a traffic sign image after grayscaling.

![alt text][image2]

As a last step, I normalized the image data to make the network faster. Without normalization, the math behind the neural network would be using significantly larger numbers and take longer to train and identify signs. 

I decided to generate additional data to even out the data set. This way, the model would not become better at identifying specific signs, but all signs. 

To add more data to the the data set, I rotated some of the new traffic signs plus or minus five degrees. Most were left un-rotated. I chose rotation because one can choose the orientation of the camera in reguard to the car and how it sits on the road, but the orientation of the signs on the side of the roads can vary depending on installation, traffic incidents, weather, and other factors. 

Here is an example of an an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is that all types of signs have a representation of at least 2000 images. This creates a fairly level playing field for the model to choose from so that it is not biased towards one sign more than others. The augmented set also contains some rotated images which I believe will help the model learn the sign shapes relationally to other parts of the sign rather than by their location in the image. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 1     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 2	 	    | 1x1 stride, same padding, outputs 10x10x16 	|
| RELU          		|           									|
| Max pooling    		| 2x2 stride,  outputs 5x5x16        			|
| Convolution 3			| 1x1 stride, same padding, outputs 1x1x400		|
| RELU          		|           									|
| Flatten       		| flatten conv 2 and 3 to 1x1x400				|
| Concatenate   		| Concatenates conv 2 and conv 3   				|
| Fully connected		| input 800, output 400							|
| RELU          		|           									|
| Dropout       		| 	        									|
| Fully connected		| input 400 output 200							|
| RELU          		|           									|
| Dropout       		|												|
| Fully connected		| input 200 output 100							|
| RELU          		|           									|
| Dropout       		|	        									|
| Fully Connected		| input 100 output 43							|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 17 epochs of 100 images with the Adam optimizer. These images were processed with a learning rate of 0.0009, dropout keep probability of 0.5, mu of 0, and sigma of 0.1. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 96.0%
* test set accuracy of 93.8%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I used the LeNet archtechture to start with, but had to modify the model due to sever underfitting. I added another comvolutional layer and a concatenation layer to give the model more data to "look at" and identify the patterns in the image. I also added more fully connected layers to give the model more chances to identify the patterns in the image. Multiple dropout layers were added to help prevent overfitting. I tuned the number of epochs, batch size, and learning rate iteratively along the process based on the results of my previous test. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

These images are fairly straighforward, clear pictures of the signs and should be relatively simple for the model to identify. These should also be well-representative of the signs encountered during normal daily driving.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead Only      		| Ahead Only   									| 
| No Entry     			| No Entry 										|
| 30 km/h				| 30 km/h										|
| Stop		      		| Stop							 				|
| Yield					| Yield      									|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.8%. The model consistently correctly guessed all five traffic signs. This is most likely due to the clarity of the images, lack of noise, lack of obstruction, lack of rotation, and relatively large size. The model may have trouble identifying signs that are further away and thus smaller, or signs that are closer and thus larger than the 32x32 input.  

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook. Below are the top 5 softmax probabilities for the extra signs plus any notes about specific signs. The model was correct every time.  

For the first image, the model is certain that this is an Ahead Only sign. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Ahead Only  									| 
| .000000000052			| Priority Road									|
| .0000000000018		| Turn Right Ahead								|
| .00000000000000041	| Yield							 				|
| .000000000000000026   | No Vehicles	      							|

For the sake of counting out 0's, I will be using scientific notation for all predictions other than the top prediction.


For the second image, the model was absolutely certain that the image was a "No Entry" sign. So much so, infact, that the other four softmax predictions are just the first four elements of the array of choices. All other signs had a probability of exactly 0.  

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No Entry   									| 
| .0					| Speed limit (20km/h)							|
| .0					| Speed limit (30km/h)							|
| .0					| Speed limit (50km/h)			 				|
| .0				    | Speed limit (60km/h) 							|

For the third image, the model is certain that this is a Speed limit (30km/h) sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (30km/h)  						| 
| 4.38918758e-13		| Speed limit (50km/h) 							|
| 1.01805866e-17		| Speed limit (70km/h)							|
| 1.98060333e-18		| Speed limit (20km/h)					 		|
| 4.41164269e-19		| Roundabout mandatory      					|

For the fourth image, the model is certain that this is a Stop sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop 		   									| 
| 1.90925101e-27		| Keep right 									|
| 4.25201096e-29		| Pedestrians									|
| 2.47386393e-29		| No entry					 					|
| 2.87039884e-30		| Speed limit (50km/h)      					|

For the fifth image, the model was absolutely certain that the image was a "Yield" sign. So much so, infact, that the other four softmax predictions are just the first four elements of the array of choices. All other signs had a probability of exactly 0.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield		  									| 
| .0					| Speed limit (20km/h)							|
| .0					| Speed limit (30km/h)							|
| .0					| Speed limit (50km/h)			 				|
| .0				    | Speed limit (60km/h) 							|



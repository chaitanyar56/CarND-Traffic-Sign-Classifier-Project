**Traffic Sign Recognition**

[//]: # (Image References)

[image1]: ./output/hist_train.jpg "Visualization Training"
[image2]: ./output/hist_valid.jpg "Visualization Validation"
[image3]: ./output/hist_test.jpg "Visualization Test"

[image4]: ./output/30_1.jpg "Grayscaling"
[image5]: ./output/30g.jpg "Original"
[image6]: ./output/my_img.jpg "Images from Internet"
[image7]: ./output/myimg.jpg "Results"


---
###Writeup
Link to my  [project code](https://github.com/chaitanyar56/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. For better designing of CNN classifier, dataset is analyzed.
* The size of training set is 34999
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

The spread of examples for each class in training, validation and test is visualized by histogram plot in the figure 1,2,3 respectively.
![alt text][image1]
![alt text][image2]
![alt text][image3]



###Design and Test a Model Architecture

####1. Preprocessing the data can reduce computations and complexity of the classifier network. Using grayscale images instead of using color can reduce the computations by 3(much helpful if if processing is done on cpu). In classification problems neural networks perform better if the data is mean centered at 0, therefore dataset is normalized before passing into neural network.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image5]
![alt text][image4]


####2. Basic LeNet architecture is chosen and output layer is modified to work with traffic sign images.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	   2x2   	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	    2x2  	| 2x2 stride,  outputs 5x5x16 				|
|	Flatten					|												|
| Fully connected	1	| Input 400  Output 120     									|
| Fully connected	1	| Input 120  Output 84     									|
| Out		| Input 84  Output 43     									|
| Softmax				|    convert Out to probabilities     									|


####3. Parameters used for training:
Learning rate = 0.001
optimizer = adam
cost function = cross entropy
Epochs= 20
Batch Size = 128

####4. Approach for choosing the model

My final model results were:
* validation set accuracy of 94.7
* test set accuracy of 93.4

LeNet is well known architecture used for digit classification and since the traffic signs are also of the same size it is a good start. From the results it can be seen that with preprocessing(grayscale and normalization) traffic signs images can result with good accuracy. No. of epochs are increased to 20 to increase the validation accuracy.

###Test a Model on New Images

####1. Performance of the network on German traffic signs from internet.

Here are  German traffic signs images  that I found on the web:

![alt text][image6]

The first image might be difficult to classify because ...

####2. Results.
The Accuracy of prediction for this images is 100%. It can be seen from the codes cells 16 from Ipython notebook.

####3. Softmax probabilities are ploted in the form of Histogram in the figure below, it can be observed that probabilities of the top class =1
![alt text][image6]

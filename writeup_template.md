# **Finding Lane Lines on the Road** 
## Writeup 

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)
[image0]: ./test_images/solidWhiteCurve.jpg "Original"
[image1]: ./GrayScale/solidWhiteCurve.jpg "Grayscale"
[image2]: ./GaussianSmoothing/solidWhiteCurve.jpg "GaussianSmoothing"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 
First, I converted the original images to **grayscale**, 


**edges = cv2.Canny(gray, low_threshold, high_threshold)**

![alt text][image0]  ![alt text][image1]

then I applied **Guassian Smoothing** to suppress noise and spurious gradients in the grayscale image using a kernel size of 5. 
![alt text][image2]

After that, to find the edges of lanes in the image, I used the OpenCV Canny function. 


In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...

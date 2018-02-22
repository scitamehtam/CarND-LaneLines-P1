# **Finding Lane Lines on the Road** 
## Writeup 

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)
[image0]: ./test_images/solidWhiteCurve.jpg "Original"
[image1]: ./GrayScale/solidWhiteCurve.jpg "Grayscale"
[image2]: ./GaussianSmoothing/solidWhiteCurve.jpg "GaussianSmoothing"
[image3]: ./Canny.solidWhiteCurve.jpg "Edges"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.
This is how our original image looks like.

![alt text][image0]

My pipeline consisted of 5 steps. 
First, I converted the original images to **grayscale**. 

**gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)**
![alt text][image1]

then I applied **Guassian Smoothing** to suppress noise and spurious gradients in the grayscale image using a kernel size of 5. 

**blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)**

![alt text][image2]

After that, to find the edges of lanes in the image, I used the OpenCV Canny function. 

**edges = cv2.Canny(blur_gray, low_threshold, high_threshold)**

I kept the low threshold as 50 and high threshold as 150.

![alt text][image3]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...

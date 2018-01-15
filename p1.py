#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
#print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image)  
# if you wanted to show a single color channel image called 'gray', 
# for example, call as plt.imshow(gray, cmap='gray')

#plt.show()

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

import os
test_images = os.listdir("test_images/")


def save_Images(outputDirectory, img, imgName, inGray=0):
	'''
	Create 'dir' directory if it doesn't exit.
	Save the 'img' image in the 'dir' directory using the 'imgName' image names. 
	save_Images(output_dir, img, imgXYZ,1) will save img in output_dir under the 
	name imgXYZ in gray 

	'''
	# Create the directory if it has not been created yet
	if not os.path.exists(outputDirectory):
		os.makedirs(outputDirectory)
	if inGray:
		mpimg.imsave(outputDirectory+'/'+imgName, img, cmap='gray')
	else :
		mpimg.imsave(outputDirectory+'/'+imgName, img)


def loop_Images(): 

    for img in test_images:
        # Read in and grayscale the image 
        image = mpimg.imread(os.path.join("test_images/",img))
        gray = grayscale(image)
        plt.imshow(gray)
        plt.show()

        #Define a kernel size and apply Gaussian smoothing and save it in another folder
        kernel_size = 5
        blur_gray = gaussian_blur(gray, kernel_size)
        save_Images('GaussianSmoothing', blur_gray, img,1)

        #Define our parameters for Canny and apply 
        low_threshold = 50 
        high_threshold = 150
        edges = canny(blur_gray, low_threshold, high_threshold)
        save_Images('Canny',edges,img,1)

        # Create a masked edge
        imshape = image.shape
        vertices = np.array([[(0,imshape[0]),(470, 320), (510, 320 ), (imshape[1],imshape[0])]], dtype=np.int32)
        masked_edges = region_of_interest(edges, vertices)

        # Define the Hough transform parameters
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hugh grid
        threshold = 15
        min_line_length = 30 
        max_line_gap = 10
        line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
        
        # Draw the lines on the edge image
        lines_edges = weighted_img(line_image, image)
        
        #Subploting all the above images in order
        fig = plt. figure()
        img1 = fig.add_subplot(161),plt.imshow(image)
        plt.title('Original')
        img2 = fig.add_subplot(162),plt.imshow(gray,cmap = 'gray')
        plt.title('Gray')
        img3 = fig.add_subplot(163),plt. imshow( blur_gray,cmap = 'gray')
        plt.title('Smoothing')
        img4 = fig.add_subplot(164),plt.imshow(edges , cmap='Greys_r')
        plt.title('Edges')
        img5 = fig.add_subplot(165),plt.imshow(line_image)
        plt.title('Hough')
        img6 = fig.add_subplot(166),plt.imshow(lines_edges)
        plt.title('Output')
        return weighted_img(hough_lines(region_of_interest(canny(gaussian_blur(grayscale(image)),kernel_size),low_threshold,high_threshold)),image)
 		
loop_Images()
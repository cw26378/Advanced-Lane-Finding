## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chess_board_undistort.png "Undistorted chessboard"
[image2]: ./output_images/road_img_undistort.png "Road Transformed"
[image3a]: ./output_images/sobel_gradient.png "Gradient Binary Example"
[image3b]: ./output_images/hls.png "HLS Binary Example"
[image4a]: ./output_images/warp_example.png "Warp Example"
[image4b]: ./output_images/warped_binary_example.png "Warped Binary Example"
[image5a]: ./output_images/sliding_window.png "Sliding Window"
[image5b]: ./output_images/fast_detect.png "Detection based on previous knowledge"
[image6]: ./output_images/result_example.png "Example of the result image"
[video1]: ./output_images/output_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

## Writeup / README

### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  
The current writeup follows the structure of the template provided by Udacity. There are 3 major parts.
1) Single Image Processing; 2) Single Image processed by the Pipeline (verification of pipeline); and 3) Video Processing based on the Image Pipeline. Most of the work is done at the single image processing part, which is the focus of this writeup as well.

As for the single image processing, there are a few key steps listed in the following:
1). Camera calibration based on the given chess board images
2). Application of Sobel threshold and HLS threshold in order to obtain binary image for lane line detection
3). Iamge Perspective Warping (to bird view) based on the given straight line image
4). Find the lane line with sliding window and then use the known window positions to search for next frame's lane line
5). Draw the lane line back onto the original image and keep the processed image


### Preparation: Camera Calibration

#### Compute the camera matrix and distortion coefficients based on the given chess board images. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code cell [2] of the IPython notebook `Advanced_LaneFinding_12-30-2017.ipynb`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objPoints` is just a replicated array of coordinates, and `objPoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgPoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objPoints` and `imgPoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  The function `cal_undistort(img, objpoints, imgpoints)` takes raw image and the objpoints and imgpoints as input, and then return a undistorted image as applying the distortion correction to the test image using the `cv2.undistort()` function. It is found out that only 17 images of the given 20 images are good for calibrations using 9 x 6 corners. So I decided to use the other three as a testing group The result of undistorting the chessboard image is this:

![alt text][image1]


### Pipeline (single images)


#### 1. Calibrate a real road image as an example of a distortion-correction.
Once the camera calibration points objPoints and imgPoints are obtained, we can apply the calibration on to the real road images.
This is done by the function `showOriginAndUndistort(img_RGB, objpoints, imgpoints)` defined in code cell[5]. And a typical result is like the following:
![alt text][image2]
Note that the change is not quite sizeable except for the areas near the edge of the camera. For example, the hood of the car looks clearly smaller after the calibration in the illustrated image. 

#### 2. Description of  using Sobel gradients combined with HLS channel thresholds to create a thresholded binary image.  

I used a combination of HLS and gradient thresholds to generate a binary image. And the code for generating gradient thesholded binary image is in code cell[7-10], while the code for generating HLS thesholded binary image is in code cell[12].  Here's an example of my output for this step.

Sobel operator threshold:
![alt text][image3a]
HLS channel threshold:
![alt text][image3b]

#### 3. Description of perspective transformation with provided example of a transformed image.

The code for my perspective transform includes a function called `warp_color(image, objPoints, imgPoints, src, dst)`, which can be found in the code cell after the threshold processes (step-3). Note that the `objPoints` and `imgPoints` are obtained from the camera calibration with chessboard images in the first step, while source and destination points of the warping `src` and `dst` are obtained manually from analyzing a known image with straigh lane lines: `/test_images/straight_lines1.jpg`. The actual points are listed in the following asa reference:

```python
height, width = image_straight_undistort.shape[:2]
src = np.float32([[581, 460],[701, 460],[299, 660],[1015, 660]])
dst = np.float32([[250,0],[1050,0],[250,height],[1050, height]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 250, 0        | 
| 701, 460      | 1050, 0       |
| 229, 660      | 250, 720      |
| 1015, 660     | 1050, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.From the image the warping of apparent traperzoid into expected rectangle box is expected, while the lane is showing curvature. Note that the warping transformation must be placed on the undistorted image, or the accuracy will be compromised.


![alt text][image4a]


Note that `warp_color` is for illustration of the perspective transformation, while the real job of warp should be done on binary image after threshold mapping/filtering. In order to better prepare for the next step of process (line finding), the function `warp_binary(binary, objpoints, imgpoints, src, dst)` is called and returns a warped binary image `top_down` or `binary_warped`, a transition matrix `M` and its inverse matrix `M_inv`. Please see the code cell [15-16] for further references.


![alt text][image4b]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
The next step is finding the lane line based on the binary image.
##### Histogram illustration 
First, we take a look at the histogram. If the binary threshold and combined channel filter works properly, the histogram will show two distinct peaks at left and right lanes respectively.
##### Sliding windows
Secondly, sliding window approach is implemented. A function named `multiWindowSlide(binary_image, n_windows)` is defined for this process. `n_windows` specifies the number of windows across the whole image. At each step of sliding window, a function `windowSlide(nonzero_pix, size_y_window, x_current, win_current, out_img)` is called to extract the indices of the non-zero pixels (portion of detected lines) in a given window together with the x coordinates. After the sliding window process, a group of (x,y) coordinates are obtained and ready for Polynomial fitting.
##### Polyfit
Thirdly, function `polyFitLine(leftx, lefty, rightx, righty)` is called to calculate the polynomially fitted curves up to n = 2.
The result of sliding window and polynomial fitting is shown in the following:
![alt text][image5a]
##### Faster detection based on the historical lane line position
After one image has managed to find the lane line, the detection for lane line in the next image can be easier thanks to the known position of previouly detected lines. Note that prec-processing such as sobel/ hls filtering are still needed to generate the binary_warped image. `fast_detect(binary_warped, margin, left_fit, right_fit)` is to detect the current binary_warped image with the help from learned parameter left_fit and right_fit with a acceptable margin. I just use `test3.jpg` as my earlier frame to see if `fast_detect` can work on `test4.jpg` as a following frame after `test3.jpg`.

![alt text][image5b]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
A function `getRadius` is defined and used for calculating the radius of curvature at each point based on the result of polynomial fitting. The offset is calculated assuming that camera is mounted at the center of car. The parameter of image scale to actual scale conversion `Meter_Per_Pix = (25.0/720, 3.0/850)` uses some realistic numbers.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

In order to deal with the sequenced data like video, a class `Line()` is defined to include the detected lane line information. Currently only one frame's information before the on-going frame is kept for fast-detection. It seems that having the earlier data is helpful, but only having one frame is insufficient. I would plan to use a queue to store a number of previous frame's knowledge of detected lines.

I implemented this step in the function `imgPipeline()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

Please find the video in the submission.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
There are a few secenarios that could be challenging for my current code. 
1) road condition change, like chaning between concrete (light brown) to pitch(dark gray) surfaces may trigger some boundary lines detected and mess up the result. I was thinking some methods with requirment of lane line slope as is used in Project-1 may still be helpful.

2) long sharp turns with changing tree shade. I noticed that at some points in my result the detected line gets unstable when both sharp turns and tree shade are combined. Personally I would agree that this could be equally challenging for human drivers. When light condition changes while road is having a sharp turn, the best thing to do is to slow down... So in reality some feedback control should be implemented when such abrupt change occurs. From the software point of view, it may be helpful if more smoothing can be used to deal with such changes. As I mentioned earlier, having multiple instead of only one history frame will help to smooth the data and make the detection more robust to noises.

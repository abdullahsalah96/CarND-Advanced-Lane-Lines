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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[image7]: ./examples/distortion_example.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the camera_calibration.py script

I implemented calibration in a class called camera which contains 3 helper functions.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. This is done in the 'get_camera_pts()' function of the class which updates both 'objpoints' and 'imgpoints'

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. This is implemented in the 'get_camera_matrix()' function which return the distortion coeffs and cam matrix to be used to undistort image

 I applied this distortion correction to the test image using the `` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)


#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

The image below shows the picture before and after using camera matrix to remove distortion
![alt text][image7]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 29 through 39 in `utils.py`). The function 'segment_lanes()' starts by getting the sobel threshold on the image from the function 'sobel_threshold()' found in line 6 then it computes the histogram threshold by calling the function 'histogram_thresholding()' in line 16, after that, color threshold is used to segment the lane lines from the image by thresholding on the L channel of the LAB color space, I chose to segment on the L channel as after trying multiple color spaces I found the L channel of the LAB color space to be the least affected by lighting. At the end the function combines all these thresholds and returns the output binary image

Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_perspective()`, which appears in lines 1 through 8 in the file `utils.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:
```python
src = np.float32([[WIDTH, HEIGHT-10],   # bottom right
                 [0, HEIGHT-10],       # bottom left
                 [546, 460],           # top left
                 [732, 460]])          # top right
dst = np.float32([[WIDTH, HEIGHT],      # bottom right
                  [0, HEIGHT],          # bottom left
                  [0, 0],               # top left
                  [WIDTH, 0]])          # top right
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 1280, 710     | 1280, 720     | 
| 0, 710        | 0, 720        |
| 546, 460      | 0, 0          |
| 732, 460      | 1280 , 0      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

This was obtained by trail as I found these values to give best result

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial in the 'lines.py' script. All these functions are encapsulated into the Lane Class found in the script

The first function of the class is the 'update_lines()' function which takes the new polynomial fits and appends them to the 'recent_lines' deque and then updates the 'last_fits' to be equal to these new fits.

The second function 'draw()' is just a helper function to draw the lines on the given mask using fill poly function

The third function 'average_fit()' which is also implemented as a property is used to compute the average of all the recent polynomial line fits.

These line fits are updated inside the 'window_search()' function found in 'utils.py' script in line 61 thorugh 156 which updates line fits and if no new lines detected it sets the current line fit to the previous one.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 49 through 63 in my code in `lines.py` script inside the Line class, The function 'curvature()' computes the curvature of the line by using the curvature formula stated in the lessons. The function 'curvature_meter()' computes the curvature in meters based on 'ym_per_pix' and 'xm_per_pix' found in the configuration parameters script 'config.py'

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 158 through 187 in my code in `utils.py` in the function `draw_lines_on_undistorted()`. it uses the inverse perspective transform which is also computed when computing the perspective transofrm and draws back lines on the image.
Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./video_example.mp4)

I used 'config.py' script to save some of the varibles used throughout the code
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. some of the problems that I faced was setting different threshold values for different videos which may cause the pipeline to fail on different lighting conditions. After doing some research, I read about a more accurate technique to segment lane lines rather than using sobel/color/histogram thresholding which is using deep learning and specifically SCNN to perform lane detection as described in this paper:
https://www.researchgate.net/figure/Comparison-between-CNN-and-SCNN-in-a-lane-detection-and-b-semantic-segmentation-For_fig2_321902196

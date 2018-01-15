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

[image1]: ./examples/undistort_failed.png "Failed undistort"
[image2]: ./examples/undistort_output.png "Undistorted"
[image3]: ./examples/undistort_test_image.png "Road Transformed"
[image4]: ./examples/filters1.png "Binary Example"
[image5]: ./examples/filters_yellow.png "Binary Example with Yellow detection"
[image6]: ./examples/parallel.png "Parallel warped"
[image7]: ./examples/lane_plot.png "Lane plotted on the road"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./Untitled.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Here I consider the number of detected objpoints and it should be equal to the expected number of objpoints to be detected (in this case 6*9=54), otherwise the transformation may be incorrect.
An example for a failed detection is:

![alt text][image1]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds similar to the ones used in the video lectures to generate a binary image (cell 4 in the ipython notebook).  Here's an example of my output for this step. 

![alt text][image4]

Then I saw that the yellow line can not be clearly detected, so I decided to add a search for yellow pixels in the Hue channel (values 18 to 22). Thus I was able to identify the yellow line while reducing the band for the saturation channel from 180:255 to 230:255. The computation time though increases by around 50% because one more pass is performed over the whole image, but for offline detection this is acceptable.

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_perspective_transform_matrix` found in the fourth cell of the notebook.  The function takes as input an image (`img`) and then returns a transformation matrix, to be used with cv2.warpPerspective() for the perspective to bird eye view and the Minverse would be used for the bird eye to perspective.  I chose the hardcode the source and destination points in the following manner:

```python
    src = np.float32([[0.16*x_size, y_size],
                         [0.5*x_size-82, 0.66*y_size],
                         [0.5*x_size+82, 0.66*y_size],
                         [0.84*x_size, y_size]])

    dst = np.float32([[0.28*x_size, y_size],
                         [0.28*x_size, 0],
                         [0.72*x_size, 0],
                         [0.72*x_size, y_size]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 205, 720      | 358, 720      | 
| 558, 475      | 358, 0      	|
| 722, 475      | 922, 0      	|
|1075, 720      | 922, 720      |

I verified that my perspective transform was working as expected by displaying warped image and checking if the warped lines are parallel and straight using the straight line example pictures.

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the fifth cell I am doing the polynomial fit as well as computing the curvature and finding the position of the vehicle in the lane. I am defining a new class called `Line` which encapsulates all the operations and data needed for drawing and computing the parameters for the lines. I also define to functions: find_warped_lines which is used to find lines when no previous data is available and fit_warped_lines which takes as input a previously found fit and the current road image. The output of both functions are the x and y coordinates of lane pixels for the left and right lines.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This is encapsulated in the Line class. By using the x and y lane coordinates for each line with the update() function we obtain:

* the 2nd order polynomial by using the np.polyfit() function
* the fitx coordinates for each y position in the warped image. Here I am using exponential smoothing with alpha = 0.2 on the previously obtained x coordinates, which should be appropriate as y stays the same.
* the curvature using the compute_curvature() function from the Line class. Here I also use exponential smoothing and I have hardcoded a cap to the turn radius at 3000m as an analogue to straight line.

The position of the vehicle and the displayed curvature are computed in the add_text() function which computes:

* the off center position as ((right fitx at the bottom - left fitx at the bottom) - 1280) / 2 and calibrates it using the meters per pixel values which in my case are approximately xm_per_pix=0.0062, ym_per_pix=0.035.
* the curvature is simply the average of the obtained curvatures for the left and right Lines. 

The add_text() function also displays the text on the image.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function draw_lane().  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output.mp4)

In the seventh cell of the notebook I have put the whole pipeline for detecting lanes in video and recording the output. I have also included all the functions used so far in a python file "helper_functions.py", so this cell can be run separately from the cells before.

Notable here is the 'sanity check' function from the Line class `is_parallel_to` which compares the obtained fitx coordinates with some hard coded values to find out if the algorithm have gone wrong, and if the hard-coded values are exceeded, we are ignoring the current input from the frame and reuse our last 'sensible' lines.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First of all using the yellow in hue channel would slow down the whole algorithm. A bettter approach should be used in real time applications.

Another problem is that my lane pixel detecting algorithm is not working well enough - it detects some pixels from the road that should not be detected (false positives) and also cannot detect some of the line pixels (false negatives). Example is when in the project video the yellow line becomes blurry, the algorithm looses track of the left line.

The third problem is the hard-coded values for sanity check which should be updated for every reposition of the camera in the car and for every change in the road we are driving on.

As a result the pipeline fails miserably on the challenging video and is barely able to keep track of the lane in the project video. Maybe a more robust approach to pixel detection would address all the three points above, or maybe they can be addressed separately as there is not a single solution to all of them - further work is required to determine this.

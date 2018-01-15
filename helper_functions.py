import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from copy import deepcopy

class Line():
    def __init__(self, y_size, xm_per_pix=0.0062, ym_per_pix=0.035):
        self.xm_per_pix = xm_per_pix
        self.ym_per_pix = ym_per_pix
        self.y_size = y_size
        self.ploty = np.linspace(0, self.y_size-1, self.y_size)
        self.detected = False
        self.x_idxs = None
        self.y_idxs = None
        self.fit = None
        self.fitx = None
        self.curvature = None
        
    def update(self, x_idxs, y_idxs):
        if len(x_idxs) > 0:
            self.x_idxs = x_idxs
            self.y_idxs = y_idxs
            self.fit = np.polyfit(y_idxs, x_idxs, 2)
            if self.detected == True:
                self.fitx = 0.8*self.fitx + 0.2*self.fit2fitx()
                self.curvature = 0.8*self.curvature + 0.2*min(self.compute_curvature(), 3000)
            else:
                self.fitx = self.fit2fitx()
                self.curvature = min(self.compute_curvature(), 3000)
            self.detected = True
        else:
            self.detected = False        
        
    def fit2fitx(self):
        fitx = self.fit[0]*self.ploty**2 + self.fit[1]*self.ploty + self.fit[2]
        return fitx
        
    def compute_curvature(self):
        # Define conversions in x and y from pixels space to meters
        y_eval = self.y_size - 1
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.ploty*self.ym_per_pix, self.fitx*self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        curverad = ((1 + (2*fit_cr[0]*y_eval*self.ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        # Now our radius of curvature is in meters
        return curverad
    
    def is_parallel_to(self, line2, min_limit_diff=500, max_limit_diff=700, limit_diff=100):
        max_diff = np.max(np.absolute(self.fitx - line2.fitx))
        min_diff = np.min(np.absolute(self.fitx - line2.fitx))                  
        parallel = (max_diff < max_limit_diff and
                    min_diff > min_limit_diff and
                    max_diff - min_diff < limit_diff)
        if not parallel:
            print (np.min(np.absolute(self.fitx - line2.fitx)), np.max(np.absolute(self.fitx - line2.fitx)))
        return parallel
    
    def copy(self):
        return deepcopy(self)

def show_binary_combination(sxbinary, s_binary, combined_binary):
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    # Plotting thresholded images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,20))
    ax1.set_title('Stacked thresholds')
    ax1.imshow(color_binary)

    ax2.set_title('Combined S channel and gradient thresholds')
    ax2.imshow(combined_binary, cmap='gray')

    plt.show()

def get_binary(im, separate_yellow_detection=True, show_plots=False):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = hls[:,:,1]

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    # Here I am using the default threshold values
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    # here I am concerned only with detecting white lines, so I can increase the minimum
    # threshhold value to 230
    s_thresh_min = 230
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    
    # Threshold by Hue channel
    # This is done to reduce the lighting range for white (less false positives)
    # and to increase detection of yellow lines which have a lower S value
    # Unfortunately this would increase the processing time by about 50% 
    # as the array should be checked one more time, and can be considered if
    # execution time is important.
    if separate_yellow_detection:
        h_channel = hls[:,:,0]
        h_thresh_min = 15
        h_thresh_max = 25
        h_binary = np.zeros_like(h_channel)
        h_binary[(h_channel >= h_thresh_min) & (h_channel <= h_thresh_max)] = 1
        combined_binary[(s_binary == 1) | (sxbinary == 1) | (h_binary == 1)] = 1
    else:
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    if show_plots:
        show_binary_combination(sxbinary, s_binary, combined_binary)
    
    return combined_binary
    
def get_perspective_transform_matrix(im):
    x_size = im.shape[1]
    y_size = im.shape[0]
    src = np.float32([[0.16*x_size, y_size],
                         [0.5*x_size-82, 0.66*y_size],
                         [0.5*x_size+82, 0.66*y_size],
                         [0.84*x_size, y_size]])

    dst = np.float32([[0.28*x_size, y_size],
                         [0.28*x_size, 0],
                         [0.72*x_size, 0],
                         [0.72*x_size, y_size]])
    M = cv2.getPerspectiveTransform(src, dst)
    return M       
    
#Reusing some of the code from 33. Finding the Lines
def find_warped_lines(warped):
    # Assuming you have created a warped binary image called "binary_warped"
    x_size = warped.shape[1]
    y_size = warped.shape[0]
    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped[y_size//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = None

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #print(leftx_base, rightx_base)

    # Choose the number of sliding windows
    nwindows = 8
    # Set height of windows
    window_height = np.int(y_size/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base    
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = y_size - (window+1)*window_height
        win_y_high = y_size - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        #print(window, (win_xleft_low,win_y_low),(win_xleft_high,win_y_high))

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
       
    return leftx, lefty, rightx, righty

def fit_warped_lines(warped, left_fit, right_fit):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty



#left_fit = np.polyfit(lefty, leftx, 2)


def fit2fitx(fit, y_size):
    ploty = np.linspace(0, y_size-1, y_size )
    fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    return left_fitx

def compute_curvature(left_fit, right_fit, y_size, xm_per_pix = 0.0062, ym_per_pix = 0.042):
    # Define conversions in x and y from pixels space to meters
    left_fitx, right_fitx, ploty = fit2fitx(left_fit, right_fit, y_size)
    
    y_eval = y_size - 1
    # Fit new polynomials to x,y in world space
    #print(ploty.shape, left_fitx.shape)
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    return left_curverad, right_curverad

# Warp the detected lane boundaries back onto the original image.
# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
# Create an image to draw the lines on
def draw_lane(left_fitx, right_fitx, M, warped_size, image_size):
    Minv = np.linalg.inv(M)
    
    ploty = np.linspace(0, image_size[0]-1, image_size[0])
    warp_zero = np.zeros(warped_size).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image_size[1], image_size[0])) 
    
    return newwarp

def add_text(img, x_pos, y_pos, left_fitx, left_curverad, right_fitx, right_curverad, xm_per_pix, ym_per_pix):
    #left_curverad, right_curverad = compute_curvature(left_fit, right_fit, img.shape[0], xm_per_pix, ym_per_pix)
    text_curverad_mean = str(round((left_curverad+right_curverad)/2, 2))+ " m radius"
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    off_center = round((((right_fitx[-1] + left_fitx[-1])-1280)/2)*xm_per_pix, 2)
    if off_center < 0:
        off_center = -off_center
        position = "right"
    else:
        position = "left"
    text_off_center = str(off_center) + " m to the " + position
    cv2.putText(img, text_curverad_mean, (x_pos, y_pos), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), thickness=4)
    cv2.putText(img, text_off_center, (x_pos, y_pos+100), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), thickness=4)
    return img

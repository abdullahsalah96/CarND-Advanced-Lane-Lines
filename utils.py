import cv2
import numpy as np
import collections
from config import *

def sobel_threshold(frame, kernel, min_threshold): 
    #perform sobel thresholding
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize= kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize= kernel)
    sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel_magnitude = np.uint8(sobel_magnitude / np.max(sobel_magnitude) * 255) #Normalization
    _, sobel_magnitude = cv2.threshold(sobel_magnitude, min_threshold, 255, cv2.THRESH_BINARY)
    return sobel_magnitude

def histogram_thresholding(frame): 
    #perform histogram_thresholding
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    _, thresholded = cv2.threshold(equalized, 200, 255,cv2.THRESH_BINARY)
    return thresholded

def lab_segmentation(frame, l_channel_min): 
    #perform color thresholding on L channel of LAB Color space as lane lines will be bright in image - affects L channel
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    thresholded = cv2.inRange(lab, (l_channel_min,0,0), (255,200,200))
    return thresholded

def segment_lanes(frame):
    #combine sobel, color and histogram thresholding to segment lane lines in image
    sobel = sobel_threshold(frame, 3, SOBEL_MIN_THRESH)
    hist = histogram_thresholding(frame)
    lab = lab_segmentation(frame, L_CHANNEL_MIN)
    cv2.imshow("hist", hist)
    cv2.imshow("lab", lab)
    cv2.imshow("sobel", sobel)
    out1 = cv2.bitwise_and(sobel, hist)
    output = cv2.bitwise_and(out1, lab)
    return output

def warp_perspective(frame, src, dst):
    #warp perspective image
    height, width = frame.shape[:2]
    if(src.all == None):
        src = np.float32([[WIDTH, HEIGHT-10],   # bottom right
                    [0, HEIGHT-10],       # bottom left
                    [546, 460],           # top left
                    [732, 460]])          # top right
    if(dst.all == None):
        dst = np.float32([[WIDTH, HEIGHT],      # bottom right
                    [0, HEIGHT],          # bottom left
                    [0, 0],               # top left
                    [WIDTH, 0]])          # top right
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(frame, M, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
    return warped, Minv



def window_search(thresh_persp, left_line, right_line, num_of_sliding_windows):
    #perform window search on perspective thresholded image
    height, width = thresh_persp.shape[:2] #shape
    histogram = np.sum(thresh_persp[height//2:-30, :], axis=0) # Take a histogram of the left half of the image
    out_img = np.dstack((thresh_persp, thresh_persp, thresh_persp)) * 255 #for visualization

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = len(histogram) // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    window_height = np.int(height / num_of_sliding_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = thresh_persp.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_base = leftx_base
    rightx_base = rightx_base

    margin = 100  # width of the windows +/- margin
    minpix = 50   # minimum number of pixels found to recenter window

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(num_of_sliding_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height
        win_xleft_low = leftx_base - margin
        win_xleft_high = leftx_base + margin
        win_xright_low = rightx_base - margin
        win_xright_high = rightx_base + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low)
                          & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low)
                           & (nonzero_x < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_base = np.int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_base = np.int(np.mean(nonzero_x[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    left_line.all_x, left_line.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
    right_line.all_x, right_line.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]
    detected = True

    if not list(left_line.all_x) or not list(left_line.all_y):
        #if no new lines detected, take last line fits
        left_fit_pixel = left_line.last_fit_pixel 
        left_fit_meter = left_line.last_fit_meter
        detected = False
    else:
        #update line fits
        left_fit_pixel = np.polyfit(left_line.all_y, left_line.all_x, 2)
        left_fit_meter = np.polyfit(left_line.all_y * ym_per_pix, left_line.all_x * xm_per_pix, 2)
        detected = True

    if not list(right_line.all_x) or not list(right_line.all_y):
        #if no new lines detected, take last line fits
        right_fit_pixel = right_line.last_fit_pixel
        right_fit_meter = right_line.last_fit_meter
        detected = False
    else:
        #update line fits
        right_fit_pixel = np.polyfit(right_line.all_y, right_line.all_x, 2)
        right_fit_meter = np.polyfit(right_line.all_y * ym_per_pix, right_line.all_x * xm_per_pix, 2)
        detected = True

    left_line.update_line(left_fit_pixel, left_fit_meter, detected=detected)
    right_line.update_line(right_fit_pixel, right_fit_meter, detected=detected)

    out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
    out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

    return left_line, right_line, out_img

def draw_lines_on_undistorted(img, inverse_perspective, left_line, right_line):
    height, width = img.shape[:2] #shape
    if(left_line.recent_lines_meter[0] is None or right_line.recent_lines_meter[0] is None):
        #No lane lines
        return img 
    else:
        left_lines = left_line.average_fit
        right_lines = right_line.average_fit
        # Generate x and y values for plotting
        ploty = np.linspace(0, height - 1, height)
        left_fitx = left_lines[0] * ploty ** 2 + left_lines[1] * ploty + left_lines[2]
        right_fitx = right_lines[0] * ploty ** 2 + right_lines[1] * ploty + right_lines[2]

        # draw road as green polygon on original frame after unwarping perspective
        road_warp = np.zeros_like(img, dtype=np.uint8) 
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(road_warp, np.int_([pts]), (0, 255, 0))
        road_dewarped = cv2.warpPerspective(road_warp, inverse_perspective, (width, height))  # Warp back to original image space
        blend_onto_road = cv2.addWeighted(img, 1., road_dewarped, 0.3, 0)
        line_warp = np.zeros_like(img)
        line_warp = left_line.draw(line_warp, color=(255, 0, 0))
        line_warp = right_line.draw(line_warp, color=(0, 0, 255))
        line_dewarped = cv2.warpPerspective(line_warp, inverse_perspective, (width, height))
        lines_mask = blend_onto_road.copy()
        idx = np.any([line_dewarped != 0][0], axis=2)
        lines_mask[idx] = line_dewarped[idx]
        blend_onto_road = cv2.addWeighted(src1=lines_mask, alpha=0.8, src2=blend_onto_road, beta=0.5, gamma=0.)
        return blend_onto_road

def offset_from_center(img, left_line, right_line):
    #calculate offset from center 
    height, width = img.shape[:2]
    if(bool(left_line.recent_lines_meter) and bool(right_line.recent_lines_meter)):
        if left_line.detected and right_line.detected:
            line_lt_bottom = np.mean(left_line.all_x[left_line.all_y > 0.65 * left_line.all_y.max()]) #get bottom of left lane line
            line_rt_bottom = np.mean(right_line.all_x[right_line.all_y > 0.65 * right_line.all_y.max()]) #get bottom of right lane line
            lane_width = line_rt_bottom - line_lt_bottom #calculate width of lane line
            midpoint = width / 2 #mid point of frame
            offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint) #offset from midpoint
            offset_meter = xm_per_pix * offset_pix #offset in meters
            # cv2.circle(img, (int(midpoint), height - 40), 10, (0,255,255), 2)  #for visualization
            # cv2.circle(img, (int(line_lt_bottom + lane_width / 2), height - 40), 10, (255,255,0), 2) #for visualization
        else:
            offset_meter = -1
    else:
        offset_meter = -1

    return offset_meter

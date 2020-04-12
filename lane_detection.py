import cv2
import numpy as np
import collections
from lines import Line
from utils import *
from config import *
from camera_calibration import Camera

video_writer = cv2.VideoWriter('video_example.mp4', -1, 20.0, (WIDTH,HEIGHT)) #video writer

################################# VIDEO 1 ####################################
cap = cv2.VideoCapture("project_video.mp4")
# cap.set(cv2.CAP_PROP_POS_MSEC,20000)      # just cue to 20 sec. position

################################# VIDEO 2 ####################################
# cap = cv2.VideoCapture("challenge_video.mp4")

################################# VIDEO 3 ####################################
# cap = cv2.VideoCapture("harder_challenge_video.mp4")

################################ Camera Calibration ###########################
cam = Camera("camera_cal/*.jpg")
mtx, dist = cam.get_camera_matrix(WIDTH, HEIGHT)

def pipeline(img):
    img = cv2.resize(img, (WIDTH, HEIGHT))      #resize all videos to one size, to avoid problem of changing code for different size videos
    img = cam.undistort_image(img, mtx, dist)   #remove distortion using camera matrix
    warped, inverse_perspective = warp_perspective(img, src, dst)   #warp image to get perspective from src to dist points (found in config.py)
    output = segment_lanes(warped)              #segment lane lines into the image - perform color and sobel thresholds to segment lane lines

    left_line = Line()                          #left lane line object
    right_line = Line()                         #right lane line object

    left_line, right_line, window_search_img = window_search(output, left_line, right_line, 4)    #perform window search to get left and right lanes and fit their positions
    line_image = draw_lines_on_undistorted(img, inverse_perspective, left_line, right_line)       #draw lines on unwarped image

    mean_curvature_meter = np.mean([left_line.curvature_meter, right_line.curvature_meter])       #calculate radius of curvature
    cv2.putText(line_image, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (100, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  
    center_offset = offset_from_center(line_image, left_line, right_line) #computer offset from center
    cv2.putText(line_image, 'Offset from Center: {:.02f}m'.format(center_offset), (100, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Window search", window_search_img)
    return line_image


############################## VIDEO #####################################
while cap.isOpened:
    _,img = cap.read()
    if FLIP:                    #bool to flip image as some local videos were flipped
        cv2.flip(img, 0, img)
    line_img = pipeline(img)
    cv2.imshow("line image", line_img)
    video_writer.write(line_img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

##########################################################################


############################## IMAGE #####################################
# img = cv2.imread("test_images/test3.jpg")
# line_img = pipeline(img)
# cv2.imwrite("image_example.jpg", line_img)

# img = cv2.imread("tst.png")
# line_img = pipeline(img)
# cv2.imshow("out", line_img)
# cv2.waitKey(0)
# cv2.imwrite("image_example.jpg", line_img)
##########################################################################
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from config import *

class Camera():
    def __init__(self, calibration_imgs_path):
        self.calibration_imgs_path = calibration_imgs_path #path of image to calibrate on
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.


    def get_camera_pts(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
        # Make a list of calibration images
        images = glob.glob(self.calibration_imgs_path)
        print(images)
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            print("Calibrating on image: " + fname)
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)        

    def get_camera_matrix(self, width, height): #function that returns camera matrix of image of given shape (width, height)
        self.get_camera_pts()
        img_size = (width, height)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size,None,None)
        return mtx, dist
 
    def undistort_image(self, img, mtx, dist): #function that returns undistorted image
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        return dst

# cam = Camera("camera_cal/*.jpg")
# mtx, dist = cam.get_camera_matrix(WIDTH, HEIGHT)
# img = cv2.imread("test_images/straight_lines1.jpg")
# undist = cam.undistort_image(img, mtx, dist)
# cv2.imwrite("undistorted.jpg", undist)
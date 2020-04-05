# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import numpy
import cv2
from aruco_board_tracker import *
import os
import pickle
import numpy as np
import pyrealsense2 as rs


# RealSense Start and extracting camera intrinsics
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

## get intrinsics
color_stream = profile.get_stream(rs.stream.color) # Fetch stream profile for depth stream
intr = color_stream.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
cameraMatrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
distCoeffs = np.zeros((1,5))
cameraMatrix
distCoeffs

ArBoard = ArucoBoardTracker(cameraMatrix, distCoeffs)


while(True):
    # Capturing each frame of our video stream
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    img = np.asanyarray(color_frame.get_data())

    if ArBoard.find_board_pose(img):
        img = ArBoard.draw_board(img)

    # Display our image
    cv2.imshow('QueryImage', img)

    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import numpy
import cv2
import cv2.aruco as aruco
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


# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_250)

# Create grid board object we're using in our stream
board = aruco.GridBoard_create(
        markersX=5,
        markersY=7,
        markerLength=0.04,
        markerSeparation=0.01,
        dictionary=ARUCO_DICT)

# Create vectors we'll be using for rotations and translations for postures
rvec, tvec = None, None

while(True):
    # Capturing each frame of our video stream
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    img = np.asanyarray(color_frame.get_data())


    # grayscale image
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect Aruco markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

    # Refine detected markers
    # Eliminates markers not part of our board, adds missing markers to the board
    corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
            image = img,
            board = board,
            detectedCorners = corners,
            detectedIds = ids,
            rejectedCorners = rejectedImgPoints,
            cameraMatrix = cameraMatrix,
            distCoeffs = distCoeffs)

    ###########################################################################
    # TODO: Add validation here to reject IDs/corners not part of a gridboard #
    ###########################################################################

    # Outline all of the markers detected in our image
    img = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 0, 255))

    # Require at least 1 marker
    if ids is not None:
        # Estimate the posture of the gridboard, which is a construction of 3D space based on the 2D video
        pose, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec, tvec)
        print(tvec[0], tvec[1], tvec[2])
        if pose:
            # Draw the camera posture calculated from the gridboard
            img = aruco.drawAxis(img, cameraMatrix, distCoeffs, rvec, tvec, 0.3)
            
    # Display our image
    cv2.imshow('QueryImage', img)

    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

'''
Contains the following classes:
   - ArucoBoardTracker - Aruco board pose estimation helper
'''


import numpy
import cv2
import cv2.aruco as aruco

class ArucoBoardTracker():
    def __init__(self, cameraMatrix, distCoeffs):
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

        self.rvec, self.tvec, self.corners = None, None, None

        self.ARUCO_PARAMETERS = aruco.DetectorParameters_create()
        self.ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.board = aruco.GridBoard_create(
            markersX=5,
            markersY=7,
            markerLength=0.04,
            markerSeparation=0.01,
            dictionary=self.ARUCO_DICT)

    def find_board_pose(self, img):
        # Detect Aruco markers
        self.corners, ids, rejectedImgPoints = aruco.detectMarkers(img, self.ARUCO_DICT, parameters=self.ARUCO_PARAMETERS)

        # Refine detected markers
        # Eliminates markers not part of our board, adds missing markers to the board
        self.corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
            image=img,
            board=self.board,
            detectedCorners=self.corners,
            detectedIds=ids,
            rejectedCorners=rejectedImgPoints,
            cameraMatrix=self.cameraMatrix,
            distCoeffs=self.distCoeffs)

        # Require at least 1 marker
        if ids is not None:
            ret, self.rvec, self.tvec = aruco.estimatePoseBoard(self.corners, ids, self.board, self.cameraMatrix, self.distCoeffs,
                                                       self.rvec, self.tvec)
        else:
            ret = False

        return ret

    def draw_board(self, img):
        img = aruco.drawDetectedMarkers(img, self.corners, borderColor=(0, 0, 255))
        # Draw the camera posture calculated from the gridboard
        img = aruco.drawAxis(img, self.cameraMatrix, self.distCoeffs, self.rvec, self.tvec, 0.3)
        return img

    def get_pose(self):
        return self.rvec, self.tvec

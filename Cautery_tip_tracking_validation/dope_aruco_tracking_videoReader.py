# This code generates the a xlsx file from a video. The code reads video file from the "output" folder, then extracts
# the location of the cautery_tip using dope and the location of the aruco board. Next, it saves those information in a
# excel file of the same name as the input video file name, in the output folder. In the excel file, two separate
# sheets are created for the two different information set.
# A separate program will be created for reading and analyzing the data.

# Usage
# Specify the "input_vid_name" variable. e.g. "input_1.avi".


import numpy as np
from aruco_board_tracker import *
from detector import *
import yaml
from PIL import Image
from PIL import ImageDraw
import pandas as pd


# Code to visualize the neural network output
def DrawLine(point1, point2, lineColor, lineWidth):
    '''Draws line on image'''
    global g_draw
    if not point1 is None and point2 is not None:
        g_draw.line([point1, point2], fill=lineColor, width=lineWidth)


def DrawDot(point, pointColor, pointRadius):
    '''Draws dot (filled circle) on image'''
    global g_draw
    if point is not None:
        xy = [
            point[0] - pointRadius,
            point[1] - pointRadius,
            point[0] + pointRadius,
            point[1] + pointRadius
        ]
        g_draw.ellipse(xy,
                       fill=pointColor,
                       outline=pointColor
                       )


def DrawCube(points, color=(255, 0, 0)):
    '''
    Draws cube with a thick solid line across
    the front top edge and an X on the top face.
    '''

    lineWidthForDrawing = 2

    # draw front
    DrawLine(points[0], points[1], color, lineWidthForDrawing)
    DrawLine(points[1], points[2], color, lineWidthForDrawing)
    DrawLine(points[3], points[2], color, lineWidthForDrawing)
    DrawLine(points[3], points[0], color, lineWidthForDrawing)

    # draw back
    DrawLine(points[4], points[5], color, lineWidthForDrawing)
    DrawLine(points[6], points[5], color, lineWidthForDrawing)
    DrawLine(points[6], points[7], color, lineWidthForDrawing)
    DrawLine(points[4], points[7], color, lineWidthForDrawing)

    # draw sides
    DrawLine(points[0], points[4], color, lineWidthForDrawing)
    DrawLine(points[7], points[3], color, lineWidthForDrawing)
    DrawLine(points[5], points[1], color, lineWidthForDrawing)
    DrawLine(points[2], points[6], color, lineWidthForDrawing)

    # draw dots
    DrawDot(points[0], pointColor=color, pointRadius=4)
    DrawDot(points[1], pointColor=color, pointRadius=4)

    # draw x on the top
    DrawLine(points[0], points[5], color, lineWidthForDrawing)
    DrawLine(points[1], points[4], color, lineWidthForDrawing)


# Extracting Dope Settings
config_name = "my_config_realsense.yaml"
yaml_path = 'cfg/{}'.format(config_name)

with open(yaml_path, 'r') as stream:
    try:
        print("Loading DOPE parameters from '{}'...".format(yaml_path))
        params = yaml.load(stream)
        print('    Parameters loaded.')
    except yaml.YAMLError as exc:
        print(exc)

    models = {}
    pnp_solvers = {}
    pub_dimension = {}
    draw_colors = {}

    # Initialize parameters
    matrix_camera = np.zeros((3, 3))
    matrix_camera[0, 0] = params["camera_settings"]['fx']
    matrix_camera[1, 1] = params["camera_settings"]['fy']
    matrix_camera[0, 2] = params["camera_settings"]['cx']
    matrix_camera[1, 2] = params["camera_settings"]['cy']
    matrix_camera[2, 2] = 1
    dist_coeffs = np.zeros((4, 1))

    if "dist_coeffs" in params["camera_settings"]:
        dist_coeffs = np.array(params["camera_settings"]['dist_coeffs'])
    config_detect = lambda: None
    config_detect.mask_edges = 1
    config_detect.mask_faces = 1
    config_detect.vertex = 1
    config_detect.threshold = 0.5
    config_detect.softmax = 1000
    config_detect.thresh_angle = params['thresh_angle']
    config_detect.thresh_map = params['thresh_map']
    config_detect.sigma = params['sigma']
    config_detect.thresh_points = params["thresh_points"]

    # For each object to detect, load network model, create PNP solver, and start ROS publishers
    for model in params['weights']:
        models[model] = \
            ModelData(
                model,
                "weights/" + params['weights'][model]
            )
        models[model].load_net_model()

        draw_colors[model] = tuple(params["draw_colors"][model])

        pnp_solvers[model] = \
            CuboidPNPSolver(
                model,
                matrix_camera,
                Cuboid3d(params['dimensions'][model], center_location=[0, -12.5, 0]),
                dist_coeffs=dist_coeffs
            )

# Setup Aruco board
ArBoard = ArucoBoardTracker(matrix_camera, np.zeros((1, 5)))

# Declaring variables for saving tracking information of both board and the tip
tip_locs = np.empty((0, 4))  # Frame number and tip locations will be saved here
board_locs = np.empty((0, 4))  # Frame number and all locations of the board will be saved here

# Reading videos for analysis
input_vid_name = "input_3.avi"
cap = cv2.VideoCapture('./output/' + input_vid_name)
frame_count = 0

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Copy and draw image
    img_copy = img.copy()
    im = Image.fromarray(img_copy)
    g_draw = ImageDraw.Draw(im)

    for m in models:
        # Detect object
        results = ObjectDetector.detect_object_in_image(
            models[m].net,
            pnp_solvers[m],
            img,
            config_detect
        )

        # Overlay cube on image
        for i_r, result in enumerate(results):
            if result["location"] is None:
                continue
            loc = result["location"]
            ori = result["quaternion"]

            # Draw the cube
            if None not in result['projected_points']:
                points2d = []
                for pair in result['projected_points']:
                    points2d.append(tuple(pair))
                DrawCube(points2d, draw_colors[m])

    # Getting the Aruco Board pose and drawing joint tracking image
    img_dope_aruco = np.array(im)
    img_dope_aruco = cv2.cvtColor(img_dope_aruco, cv2.COLOR_RGB2BGR)
    ArBoard_pose_success = ArBoard.find_board_pose(img)
    if ArBoard_pose_success:
        img_dope_aruco = ArBoard.draw_board(img_dope_aruco)
        board_rvec, board_tvec = ArBoard.get_pose()
    cv2.imshow('DOPE Output', img_dope_aruco)

    # Printing and saving Tip and Board locations
    for i_r, result in enumerate(results):
        if result["location"] is None:
            continue
        loc = np.array(result["location"])
        print("Pen tip:      " + str(loc))
        new_row = np.insert(loc.flatten(), 0, frame_count)
        tip_locs = np.append(tip_locs, new_row.reshape(1, -1), axis=0)

    if ArBoard_pose_success:
        print("Board Center: " + str(
            np.squeeze(board_tvec) * 100))  # expressing it as a 1 dimensional array with cm unit
        print("\n")
        new_row = np.insert(np.squeeze(board_tvec) * 100, 0, frame_count)
        board_locs = np.append(board_locs, new_row.reshape(1, -1), axis=0)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # t_end = time.time()
    # print(1/(t_end-t_start))
cv2.destroyAllWindows()
cap.release()

# Saving data in a csv file
tip_locs = pd.DataFrame(data=tip_locs, columns=["Frame", "x", "y", "z"])
board_locs = pd.DataFrame(data=board_locs, columns=["Frame", "x", "y", "z"])

excel_file_dir = "./output/" + input_vid_name.split(".")[0] + ".xlsx"

with pd.ExcelWriter(excel_file_dir) as writer:
    tip_locs.to_excel(writer, index=False, sheet_name="tip_locs")
    board_locs.to_excel(writer, index=False, sheet_name="board_locs")

print("done")

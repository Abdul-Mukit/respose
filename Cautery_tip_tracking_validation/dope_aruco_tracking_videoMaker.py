'''
Use this code to make raw videos for tracking validation. The code uses intel realsense D435i camera. Two videos are
saved in the raw_videos folder. One is the raw video and another is the dope-aruco tracked video. Later these videos
are used in video editor to produce the "ouput" folder's videos which are used for the tracking excel file generation.

Usage:
just specify "file_numb" variable. Give a new number every time you collect a new video.
'''

import numpy as np
from aruco_board_tracker import *
from detector import *
import yaml

import pyrealsense2 as rs

from PIL import Image
from PIL import ImageDraw


### Code to visualize the neural network output

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


# Settings
config_name = "my_config_realsense.yaml"
exposure_val = 166


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
    matrix_camera = np.zeros((3,3))
    matrix_camera[0,0] = params["camera_settings"]['fx']
    matrix_camera[1,1] = params["camera_settings"]['fy']
    matrix_camera[0,2] = params["camera_settings"]['cx']
    matrix_camera[1,2] = params["camera_settings"]['cy']
    matrix_camera[2,2] = 1
    dist_coeffs = np.zeros((4,1))

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
                Cuboid3d(params['dimensions'][model]),
                dist_coeffs=dist_coeffs
            )

# Setup Aruco board
ArBoard = ArucoBoardTracker(matrix_camera, np.zeros((1,5)))

# RealSense Start
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
# Setting exposure
s = profile.get_device().query_sensors()[1]
s.set_option(rs.option.exposure, exposure_val)

# Define the codec and create VideoWriter object
file_numb = 4
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out_folder_dir = "raw_videos"
orv_vid_name = "./" + out_folder_dir + "/input_" + str(file_numb) + ".avi"
dope_aruco_vid_name = "./" + out_folder_dir + "/dope_aruco_annotated_" + str(file_numb) + ".avi"

out_vid_org = cv2.VideoWriter(orv_vid_name, fourcc, 30.0, (640, 480))
out_vid_dope_aruco = cv2.VideoWriter(dope_aruco_vid_name, fourcc, 30.0, (640, 480))

while True:
    # Reading image from camera
    # t_start = time.time()
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    img = np.asanyarray(color_frame.get_data())
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

    img_dope_aruco = np.array(im)
    img_dope_aruco = cv2.cvtColor(img_dope_aruco, cv2.COLOR_RGB2BGR)

    # Getting the Aruco Board pose and drawing image
    if ArBoard.find_board_pose(img):
        img_dope_aruco = ArBoard.draw_board(img_dope_aruco)

    # Displaying the dope-aruco overlaid image
    cv2.imshow('DOPE Output', img_dope_aruco)

    # Saving both dope-aruco and original video streams
    out_vid_org.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    out_vid_dope_aruco.write(img_dope_aruco)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # t_end = time.time()
    # print(1/(t_end-t_start))
cv2.destroyAllWindows()
out_vid_dope_aruco.release()
out_vid_org.release()






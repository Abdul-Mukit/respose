from detector import *
import yaml
import time
from dope_utilities import *
from server import DopeServer

# Settings
cuda_device = 0 # device to run demo on
network = "DOPE" # Select between "DOPE" and "ResPose" to run.
config_name = "my_config_webcam.yaml"
dope_server = DopeServer(SERVER_IP='127.0.0.1', PORT_NUMBER=54000, SIZE=256)

yaml_path = 'cfg/{}'.format(config_name)
with open(yaml_path, 'r') as stream:
    try:
        print("Loading DOPE parameters from '{}'...".format(yaml_path))
        params = yaml.load(stream, Loader=yaml.FullLoader)
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
            ModelData(model,
                      "weights/" + params['weights'][model],
                      cuda_device,
                      network
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

cap = cv2.VideoCapture(0)

while True:
    # Reading image from camera
    t_start = time.time()
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Copy and draw image
    img_copy = img.copy()
    im = Image.fromarray(img_copy)
    g_draw = ImageDraw.Draw(im)

    for m in models:
        # Detect object
        t_start_dope = time.time()
        results = ObjectDetector.detect_object_in_image(
            models[m].net,
            network,
            pnp_solvers[m],
            img,
            config_detect,
            cuda_device
        )
        t_end_dope = time.time()
        # Overlay cube on image
        for i_r, result in enumerate(results):
            if result["location"] is None:
                continue
            loc = result["location"]
            ori = result["quaternion"]

            # Sending to unity client
            loc = np.array(loc)/10 # cm to meter conversion
            loc = loc.tolist()
            dope_server.send(loc + ori.tolist())

            # Draw the cube
            if None not in result['projected_points']:
                points2d = []
                for pair in result['projected_points']:
                    points2d.append(tuple(pair))
                DrawCube(points2d, draw_colors[m], draw=g_draw)

    open_cv_image = np.array(im)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    cv2.imshow('Open_cv_image', open_cv_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    t_end = time.time()
    print('Overall FPS: {}, DOPE fps: {}'.format(1 / (t_end - t_start), 1 / (t_end_dope - t_start_dope)))
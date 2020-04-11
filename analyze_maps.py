import cv2
import matplotlib.pyplot as plt
import torch
from detector import *

## Settings
name = 'meat'
net_path = 'weights/dope_meat_v1/net_dope_meat_v1_60.pth'
# net_path = 'weights/rp_meat_v1/net_rp_meat_v1_60.pth'
gpu_id = 0
img_path = 'Dataset/dev/000000.left.jpg'
network="DOPE"

# def normalize_maps(activations):



# Function for visualizing feature maps
def viz_activation_maps(activations):
    fig = plt.figure(figsize=(20, 20))
    row = 1
    n_filters = activations.shape[0]
    for i in range(n_filters):
        ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
        # grab layer outputs
        ax.imshow(np.squeeze(activations[i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i + 1))

# load color image
in_img = cv2.imread(img_path)
in_img = cv2.resize(in_img, (640, 480))
in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
# plot image
plt.imshow(in_img)


model = ModelData(name, net_path, gpu_id, network)
model.load_net_model()
net_model = model.net

# Run network inference
image_tensor = transform(in_img)
image_torch = Variable(image_tensor).cuda().unsqueeze(0)
out, seg = net_model(image_torch)
beliefs = out[-1][0].cpu() # Select the last cascade's output only
affinities = seg[-1][0].cpu() # Select the last cascade's output only

# View the vertex and affinities
viz_activation_maps(beliefs)
viz_activation_maps(affinities)

plt.show()


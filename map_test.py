from dope_utilities import *
import matplotlib.pyplot as plt
import numpy as np


def get_truth_maps(data_path, index, sigma=8):
    """
    :param
    data_path: Folder path where data is loaded
    index: index of the image we want to test
    sigma: how big should be the belief points in maps

    :return:
    beliefsImg: as list of Image objects
    affinities: tensor array
    """
    imgs_data = loadimages(data_path)
    path, name, txt = imgs_data[index]
    img = default_loader(path)
    data = loadjson(path=txt, objectsofinterest=None, img=img)
    pointsBelief = data['pointsBelief']
    objects_centroid = data['centroids']
    beliefsImg = CreateBeliefMap(img, pointsBelief=pointsBelief, nbpoints=9, sigma=sigma)
    affinities = GenerateMapAffinity(img, 8, pointsBelief, objects_centroid, scale=1)
    img = np.array(img)
    return img, beliefsImg, affinities


path = "Dataset/dev_batch/"
i = 1
img, beliefsImg, affinities = get_truth_maps(path, i)
map = np.array(beliefsImg[0])

map = cv2.cvtColor(map, cv2.COLOR_RGBA2GRAY)
print(map.shape)


# Create mask out of the belief map
threshold = 0.9
map -= map.min()
map = map/map.max()
# map = map.data.numpy()
map = cv2.resize(map, (height,width))
map[map>=threshold] = 1
map[map<threshold] = 0


# plt.imshow(img)
# # plt.imshow(bImg)
# plt.show()
# print("hoasi")
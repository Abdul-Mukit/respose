# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

'''
Contains the following classes:
   - ModelData - High level information encapsulation
   - ObjectDetector - Greedy algorithm to build cuboids from belief maps 
'''

import time
import json
import os, shutil
import sys
import traceback
from os import path
import threading
from threading import Thread
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from scipy import ndimage
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage.filters import gaussian_filter
from networks import *
from networks_exp import *
# Import the definition of the neural network model and cuboids
from cuboid_pnp_solver import *
from dope_utilities import *

#global transform for image input
transform = transforms.Compose([
    # transforms.Scale(IMAGE_SIZE),
    # transforms.CenterCrop((imagesize,imagesize)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

class ModelData(object):
    '''This class contains methods for loading the neural network'''

    def __init__(self, name="", net_path="", gpu_id=0, network="DOPE"):
        self.name = name
        self.net_path = net_path  # Path to trained network model
        self.net = None  # Trained network
        self.gpu_id = gpu_id
        self.network = network

    def get_net(self):
        '''Returns network'''
        if not self.net:
            self.load_net_model()
        return self.net

    def load_net_model(self):
        '''Loads network model from disk'''
        if not self.net and path.exists(self.net_path):
            self.net = self.load_net_model_path(self.net_path)
        if not path.exists(self.net_path):
            print("ERROR:  Unable to find model weights: '{}'".format(
                self.net_path))
            exit(0)

    def load_net_model_path(self, path):
        '''Loads network model from disk with given path'''
        model_loading_start_time = time.time()
        print("Loading model '{}'...".format(path))
        device = torch.device("cuda:" + str(self.gpu_id))

        if self.network == "DOPE": # TODO: Check whether network selection works
            net = DopeNetwork()
        elif self.network == "DOPE_2":
            net = DOPE_2()
        elif self.network == "DOPE_2.1":
            net = DOPE_2p1()
        elif self.network == "DOPE_2.2":
            net = DOPE_2p2()
        elif self.network == "ResNetPose":
            net = ResNetPose()

        net = net.to(device)  # For model not trained with dataparallel
        net.load_state_dict(torch.load(path))
        net.eval()
        print('    Model loaded in {} seconds.'.format(
            time.time() - model_loading_start_time))
        return net

    # def load_net_model_path(self, path):
    #     '''Loads network model from disk with given path'''
    #     model_loading_start_time = time.time()
    #     print("Loading DOPE model '{}'...".format(path))
    #     device = torch.device("cuda:0")
    #     net = ResPoseNetwork()
    #     net = net.to(device)  # For model not trained with dataparallel
    #     try:
    #         net = net.to(device)  # For model not trained with dataparallel
    #         net.load_state_dict(torch.load(path))
    #     except:
    #         net = torch.nn.DataParallel(net, [0]).cuda()  # For model trained with dataparallel
    #         net.load_state_dict(torch.load(path))
    #
    #     net.eval()
    #     print('    Model loaded in {} seconds.'.format(
    #         time.time() - model_loading_start_time))
    #     return net

    def __str__(self):
        '''Converts to string'''
        return "{}: {}".format(self.name, self.net_path)


#================================ ObjectDetector ================================
class ObjectDetector(object):
    '''This class contains methods for object detection'''

    @staticmethod
    def detect_object_in_image(net_model, network, pnp_solver, in_img, config, gpu_id=0):
        '''Detect objects in a image using a specific trained network model'''

        if in_img is None:
            return []

        # Run network inference
        image_tensor = transform(in_img)
        device = torch.device("cuda:" + str(gpu_id))
        image_torch = Variable(image_tensor).to(device).unsqueeze(0)
        # out, seg = net_model(image_torch)

        if network == "DOPE":
            out, seg = net_model(image_torch)
        else:
            out, seg = reshape_maps(net_model(image_torch))  # shape of mapList is different from DOPE's

        vertex2 = out[-1][0]
        aff = seg[-1][0]

        # Find objects from network output
        detected_objects = ObjectDetector.find_object_poses(vertex2, aff, pnp_solver, config)

        return detected_objects

    @staticmethod
    def find_object_poses(vertex2, aff, pnp_solver, config):
        '''Detect objects given network output'''

        # Detect objects from belief maps and affinities
        objects, all_peaks = ObjectDetector.find_objects(vertex2, aff, config)
        detected_objects = []
        obj_name = pnp_solver.object_name

        for obj in objects:
            # Run PNP
            points = obj[1] + [(obj[0][0]*8, obj[0][1]*8)]
            cuboid2d = np.copy(points)
            location, quaternion, projected_points = pnp_solver.solve_pnp(points)

            # Save results
            detected_objects.append({
                'name': obj_name,
                'location': location,
                'quaternion': quaternion,
                'cuboid2d': cuboid2d,
                'projected_points': projected_points,
            })

        return detected_objects

    @staticmethod
    def find_objects(vertex2, aff, config, numvertex=8):
        '''Detects objects given network belief maps and affinities, using heuristic method'''

        all_peaks = []
        peak_counter = 0
        for j in range(vertex2.size()[0]):
            belief = vertex2[j].clone()
            map_ori = belief.cpu().data.numpy()
            
            map = gaussian_filter(belief.cpu().data.numpy(), sigma=config.sigma)
            p = 1
            map_left = np.zeros(map.shape)
            map_left[p:,:] = map[:-p,:]
            map_right = np.zeros(map.shape)
            map_right[:-p,:] = map[p:,:]
            map_up = np.zeros(map.shape)
            map_up[:,p:] = map[:,:-p]
            map_down = np.zeros(map.shape)
            map_down[:,:-p] = map[:,p:]

            peaks_binary = np.logical_and.reduce(
                                (
                                    map >= map_left, 
                                    map >= map_right, 
                                    map >= map_up, 
                                    map >= map_down, 
                                    map > config.thresh_map)
                                )
            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) 
            
            # Computing the weigthed average for localizing the peaks
            peaks = list(peaks)
            win = 5
            ran = win // 2
            peaks_avg = []
            for p_value in range(len(peaks)):
                p = peaks[p_value]
                weights = np.zeros((win,win))
                i_values = np.zeros((win,win))
                j_values = np.zeros((win,win))
                for i in range(-ran,ran+1):
                    for j in range(-ran,ran+1):
                        if p[1]+i < 0 \
                                or p[1]+i >= map_ori.shape[0] \
                                or p[0]+j < 0 \
                                or p[0]+j >= map_ori.shape[1]:
                            continue 

                        i_values[j+ran, i+ran] = p[1] + i
                        j_values[j+ran, i+ran] = p[0] + j

                        weights[j+ran, i+ran] = (map_ori[p[1]+i, p[0]+j])

                # if the weights are all zeros
                # then add the none continuous points
                OFFSET_DUE_TO_UPSAMPLING = 0.4395
                try:
                    peaks_avg.append(
                        (np.average(j_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING, \
                         np.average(i_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING))
                except:
                    peaks_avg.append((p[0] + OFFSET_DUE_TO_UPSAMPLING, p[1] + OFFSET_DUE_TO_UPSAMPLING))
            # Note: Python3 doesn't support len for zip object
            peaks_len = min(len(np.nonzero(peaks_binary)[1]), len(np.nonzero(peaks_binary)[0]))

            peaks_with_score = [peaks_avg[x_] + (map_ori[peaks[x_][1],peaks[x_][0]],) for x_ in range(len(peaks))]

            id = range(peak_counter, peak_counter + peaks_len)

            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += peaks_len

        objects = []

        # Check object centroid and build the objects if the centroid is found
        for nb_object in range(len(all_peaks[-1])):
            if all_peaks[-1][nb_object][2] > config.thresh_points:
                objects.append([
                    [all_peaks[-1][nb_object][:2][0],all_peaks[-1][nb_object][:2][1]],
                    [None for i in range(numvertex)],
                    [None for i in range(numvertex)],
                    all_peaks[-1][nb_object][2]
                ])

        # Working with an output that only has belief maps
        if aff is None:
            if len (objects) > 0 and len(all_peaks)>0 and len(all_peaks[0])>0:
                for i_points in range(8):
                    if  len(all_peaks[i_points])>0 and all_peaks[i_points][0][2] > config.threshold:
                        objects[0][1][i_points] = (all_peaks[i_points][0][0], all_peaks[i_points][0][1])
        else:
            # For all points found
            for i_lists in range(len(all_peaks[:-1])):
                lists = all_peaks[i_lists]

                for candidate in lists:
                    if candidate[2] < config.thresh_points:
                        continue

                    i_best = -1
                    best_dist = 10000 
                    best_angle = 100
                    for i_obj in range(len(objects)):
                        center = [objects[i_obj][0][0], objects[i_obj][0][1]]

                        # integer is used to look into the affinity map, 
                        # but the float version is used to run 
                        point_int = [int(candidate[0]), int(candidate[1])]
                        point = [candidate[0], candidate[1]]

                        # look at the distance to the vector field.
                        v_aff = np.array([
                                        aff[i_lists*2, 
                                        point_int[1],
                                        point_int[0]].data.item(),
                                        aff[i_lists*2+1, 
                                            point_int[1], 
                                            point_int[0]].data.item()]) * 10

                        # normalize the vector
                        xvec = v_aff[0]
                        yvec = v_aff[1]

                        norms = np.sqrt(xvec * xvec + yvec * yvec)

                        xvec/=norms
                        yvec/=norms
                            
                        v_aff = np.concatenate([[xvec],[yvec]])

                        v_center = np.array(center) - np.array(point)
                        xvec = v_center[0]
                        yvec = v_center[1]

                        norms = np.sqrt(xvec * xvec + yvec * yvec)
                            
                        xvec /= norms
                        yvec /= norms

                        v_center = np.concatenate([[xvec],[yvec]])
                        
                        # vector affinity
                        dist_angle = np.linalg.norm(v_center - v_aff)

                        # distance between vertexes
                        dist_point = np.linalg.norm(np.array(point) - np.array(center))
                        
                        if dist_angle < config.thresh_angle \
                                and best_dist > 1000 \
                                or dist_angle < config.thresh_angle \
                                and best_dist > dist_point:
                            i_best = i_obj
                            best_angle = dist_angle
                            best_dist = dist_point

                    if i_best is -1:
                        continue
                    
                    if objects[i_best][1][i_lists] is None \
                            or best_angle < config.thresh_angle \
                            and best_dist < objects[i_best][2][i_lists][1]:
                        objects[i_best][1][i_lists] = ((candidate[0])*8, (candidate[1])*8)
                        objects[i_best][2][i_lists] = (best_angle, best_dist)

        return objects, all_peaks

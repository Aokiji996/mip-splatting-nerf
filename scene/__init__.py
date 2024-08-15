#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import socket
import numpy as np
import cv2
import struct
from convert_blender_data import convert_to_nerfdata_from_socket
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "metadata.json")):
            print("Found metadata.json file, assuming multi scale Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Multi-scale"](args.source_path, args.white_background, args.eval, args.load_allres)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

def recvall(sock, count):
    """ Helper function to ensure we receive exactly the amount of bytes we expect """
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def mat_type_to_dtype(mat_type):
    """ Convert OpenCV Mat type to numpy dtype """
    # Constants for the depth
    depth_to_dtype = {
        cv2.CV_8U: np.uint8,
        cv2.CV_8S: np.int8,
        cv2.CV_16U: np.uint16,
        cv2.CV_16S: np.int16,
        cv2.CV_32S: np.int32,
        cv2.CV_32F: np.float32,
        cv2.CV_64F: np.float64,
    }
    depth = mat_type & 7  # Last 3 bits of the type
    dtype = depth_to_dtype.get(depth, None)
    if dtype is None:
        raise ValueError("Unsupported OpenCV depth: {}".format(depth))
    return dtype

def receive_file(sock, filename):
    with open(filename, 'wb') as file:
        # Receive the data length (uint32_t)
        raw_length = recvall(sock, 4)
        if not raw_length:
            return  # No more data, exit loop
        data_length = struct.unpack('I', raw_length)[0]
        if data_length == 0:
            return  # Zero length indicates end of transmission

        print("Receiving file: {} ({} bytes)".format(filename, data_length))
        # Receive and write the file data
        file_data = b''
        while len(file_data) < data_length:
            packet = sock.recv(min(data_length - len(file_data), 10 * 1024 * 1024))
            if not packet:
                break  # No more data, exit loop
            file_data += packet

        file.write(file_data)
        print('Received point3d.ply')


def receive_data(sock, duration= 1):
    frames = {
        'focal': [],
        'cam2world': [],
        'pix2cam': [],
        'img': []
    }
    count = 0
    while True:
        # Receive the data length (uint32_t)
        raw_length = recvall(sock, 4)
        if not raw_length:
            break  # No more data, exit loop
        data_length = struct.unpack('I', raw_length)[0]
        if data_length == 0:
            break  # Zero length indicates end of transmission

        # Continue to receive and deserialize the data as before
        focal = struct.unpack('f', recvall(sock, 4))[0]
        cam2world = np.frombuffer(recvall(sock, 4*16), dtype=np.float32).reshape((4, 4))
        pix2cam = np.frombuffer(recvall(sock, 4*9), dtype=np.float32).reshape((3, 3))

        rows = struct.unpack('i', recvall(sock, 4))[0]
        cols = struct.unpack('i', recvall(sock, 4))[0]
        img_type = struct.unpack('i', recvall(sock, 4))[0]
        dtype = mat_type_to_dtype(img_type)
        channels = 1 + (img_type >> 3)  # Higher bytes represent number of channels minus one
        img_data = recvall(sock, rows * cols * channels * np.dtype(dtype).itemsize)
        img = np.frombuffer(img_data, dtype=dtype).reshape((rows, cols, channels))
        
        if count % duration == 0:
            frames['focal'].append(focal)
            frames['cam2world'].append(cam2world)
            frames['pix2cam'].append(pix2cam)
            frames['img'].append(img)
        
        count += 1
        print("Received {} frames".format(count))

    return frames

class SocketScene:

    gaussians : GaussianModel
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], host="127.0.0.1", port=12345, duration= 4):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        self.train_cameras = {}
        self.test_cameras = {}
        # 创建套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 绑定地址和端口
        sock.bind((host, port))  # 设置正确的地址和端口号

        # 监听连接
        sock.listen(1)

        # 等待连接
        print("Waiting for connection...")
        # Before starting to listen for connections
        # with open('/tmp/python_ready.lock', 'w') as lock_file:
        #     lock_file.write('ready')    
        conn, addr = sock.accept()
        print("Connected!")

        frames = receive_data(conn, duration)
        print("Received {} frames".format(len(frames['focal'])))
        receive_file(conn, 'points3d.ply')
        # 关闭连接
        conn.close()
        sock.close()
        ndown = 1
        meta_with_imgs = convert_to_nerfdata_from_socket(frames, ndown)
        scene_info = sceneLoadTypeCallbacks["Multi-scale-Socket"](meta_with_imgs, args.white_background, args.eval, args.load_allres, ndown)

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

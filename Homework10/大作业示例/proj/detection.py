import clustering
import os
import numpy as np
import struct
import open3d as o3d
from mypointnet.src.model import PointNet
import torch
import re
import math


def load_ckp(ckp_path, model):
  state = torch.load(ckp_path)
  model.load_state_dict(state['state_dict'])
  print("model load from %s" % ckp_path)


def show_np_pts3d(np_pts):
    pc_view = o3d.geometry.PointCloud()
    pc_view.points = o3d.utility.Vector3dVector(np_pts)
    o3d.visualization.draw_geometries([pc_view])


def read_velodyne_bin(path):
  '''
  :param path:
  :return: homography matrix of the point cloud, N*3
  '''
  pc_list = []
  with open(path, 'rb') as f:
    content = f.read()
    pc_iter = struct.iter_unpack('ffff', content)
    for idx, point in enumerate(pc_iter):
      pc_list.append([point[0], point[1], point[2]])
  return np.asarray(pc_list, dtype=np.float32)


def process_pcd_files(dir):
  label_dir = dir + "/label_2"
  pcd_dir = dir + "/velodyne"
  calib_dir = dir + "/calib"
  pdc_files = []
  label_files = []
  calib_files = []
  things = os.listdir(pcd_dir)
  for f in things:
    pdc_file = pcd_dir + '/' + f
    if not os.path.isdir(pdc_file):
      pdc_files.append(pdc_file)
      label_file = label_dir + '/' + f.split(".bin")[0] + ".txt"
      calib_file = calib_dir + '/' + f.split(".bin")[0] + ".txt"
      label_files.append(label_file)
      calib_files.append(calib_file)
      # break
  return pdc_files, label_files, calib_files


def extracted_points(np_pts):
  left_points, gd_vec = clustering.ground_segmentation(np_pts)
  # print("clustering...")
  cluster_index, pca_vec = clustering.clustering(left_points)
  assert(len(cluster_index) == left_points.shape[0])
  # print("extract genobj...")
  clusters = {}
  for i in range(left_points.shape[0]):
    if cluster_index[i] in clusters:
      clusters[cluster_index[i]].append(i)
    else:
      clusters[cluster_index[i]] = [i]
  
  obstacles = []
  for _, pts_ids in clusters.items():
    pts = left_points[pts_ids]
    obstacles.append(pts)
  
  return obstacles, gd_vec, pca_vec


def preprocess(feature):
  # normalize
  center = np.expand_dims(np.mean(feature, axis = 0), 0)
  feature = feature - center
  dist = np.max(np.sqrt(np.sum(feature ** 2, axis = 1)),0)
  feature = feature / dist #scale
  feature = np.expand_dims(feature.T, axis=0) # mock batch size
  print("preprocess shape: ", feature.shape)
  feature = torch.Tensor(feature)
  return feature


def get_size(np_pts, gd_vec, pca_vec):
  # height
  height = 0.0
  [nor_vec_final, pt_final] = gd_vec
  for i in range(np_pts.shape[0]):
    pt = np_pts[i]
    dist = math.fabs(np.matmul(nor_vec_final, (pt - pt_final).T) / np.linalg.norm(nor_vec_final))
    height = max(height, dist)

  # width, length
  low_dim_data = np_pts * pca_vec
  x_min = float("inf")
  x_max = -float("inf")
  y_min = float("inf")
  y_max = -float("inf")
  for i in range(low_dim_data.shape[0]):
      pt = low_dim_data[i]
      x_min = min(x_min, pt[0, 0])
      x_max = max(x_max, pt[0, 0])
      y_min = min(y_min, pt[0, 1])
      y_max = max(y_max, pt[0, 1])
  width = min(y_max - y_min, x_max - x_min)
  length = max(y_max - y_min, x_max - x_min)
  return height, width, length


def get_position(np_pts):
  mean = np.mean(np_pts, axis=0)
  return mean[0], mean[1], mean[2]


if __name__ == "__main__":
  gpus = [0]
  batch_size = 1
  ckp_path = './mypointnet/output/latest.pth'
  device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
  model = PointNet().to(device=device)
  if ckp_path:
    load_ckp(ckp_path, model)
    model = model.to(device)
  
  model.eval()

  pcd_files, label_files, calib_files = process_pcd_files("../dataset/training")
  dt_dir = "./dt/"
  for pcd_file in pcd_files:
    res = re.findall("../dataset/training/velodyne/(.*?).bin", pcd_file)
    dt_file = dt_dir + res[0] + ".txt"
    print(dt_file)
    np_pts = read_velodyne_bin(pcd_file)
    obstacles, gd_vec, pca_vec = extracted_points(np_pts)
    for obstacle in obstacles:
      with torch.no_grad():
        feature = preprocess(obstacle)
        feature = feature.to(device)
        prediction = model(feature)
        pred_y = np.argmax(prediction.cpu().numpy(), axis=1)
        classes = ['Car', 'Pedestrian', 'Cyclist', 'genobj']
        # print("This pile of pointclouds is predicted to be[" + classes[pred_y[0]] + "]")
        # show_np_pts3d(obstacle)
        if classes[pred_y[0]] != 'genobj':
          height, width, length = get_size(obstacle, gd_vec, pca_vec)
          x, y, z = get_position(obstacle)
          if height <= 0.5 or width * length < 0.04:
            continue
          with open(dt_file, 'a') as fw:
            line = classes[pred_y[0]] + " 0.00 0 0.00 600.0 170.0 630.0 201.0 " + \
              str(height) + " " + str(width) + " " + str(length) + " " + str(x) + \
                " " + str(y) + " " + str(z) + " 0.00 1.0"
            fw.write(line + '\n')
            print("write " + dt_file)
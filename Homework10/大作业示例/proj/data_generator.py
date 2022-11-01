import numpy as np
import os
from third_lib.utils import calibration_kitti
from third_lib.utils import box_utils
import open3d as o3d
import struct
import json
from tqdm import tqdm
import time
import clustering


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
  return pdc_files, label_files, calib_files


def process_label_file(label_file):
  label_infos = []
  with open(label_file, 'r') as f:
    for line in f:
      info = list(map(str, line[:-1].split(' ')))
      label_info = {}
      label_info['class'] = info[0]
      label_info['height'] = info[8]
      label_info['width'] = info[9]
      label_info['length'] = info[10]
      label_info['x'] = info[11]
      label_info['y'] = info[12]
      label_info['z'] = info[13]
      label_info['theta'] = info[14]
      if label_info['class'] == "DontCare":
        continue
      label_infos.append(label_info)
      # print(label_info)
  return label_infos


def extracted_points(np_pts, label_infos, calibration):
  samples = []
  # extract ped, car, cyclist
  # for info in label_infos:
  #   gt_boxes_camera = np.array([[info['x'], info['y'], info['z'], info['length'], info['height'], info['width'], info['theta']]]).astype(np.float32)
  #   gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calibration)
  #   # print(gt_boxes_lidar[0])
  #   info['box_lidar'] = box_utils.boxes_to_corners_3d(gt_boxes_lidar)[0]
  #   is_in_box = box_utils.in_hull(np_pts, info['box_lidar'])
  #   pts = []
  #   for i in range(np_pts.shape[0]):
  #     if is_in_box[i]:
  #       pts.append(np_pts[i])
  #   if len(pts) < 20:
  #     continue
  #   samples.append({'class': info['class'], 'pts': np.array(pts)})
    # show_np_pts3d(samples[-1]['pts'])
  
  # extract genobj
  # print("ground segmentation...")
  left_points = clustering.ground_segmentation(np_pts)
  # print("clustering...")
  cluster_index = clustering.clustering(left_points)
  assert(len(cluster_index) == left_points.shape[0])
  # print("extract genobj...")
  clusters = {}
  for i in range(left_points.shape[0]):
    if cluster_index[i] in clusters:
      clusters[cluster_index[i]].append(i)
    else:
      clusters[cluster_index[i]] = [i]
  max_num_one_file = 10
  for _, pts_ids in clusters.items():
    pts = left_points[pts_ids]
    if pts.shape[0] < 20:
      continue
    # show_np_pts3d(pts)
    is_genobj = True
    for info in label_infos:
      gt_boxes_camera = np.array([[info['x'], info['y'], info['z'], info['length'], info['height'], info['width'], info['theta']]]).astype(np.float32)
      gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calibration)
      info['box_lidar'] = box_utils.boxes_to_corners_3d(gt_boxes_lidar)[0]
      is_in_box = box_utils.in_hull(pts, info['box_lidar'])
      for j in range(is_in_box.shape[0]):
        if is_in_box[j] == True:
          is_genobj = False
          break
      if not is_genobj:
        break
    if is_genobj:
      samples.append({'class': 'genobj', 'pts': pts})
      max_num_one_file -= 1
    if max_num_one_file <= 0:
      break
  return samples


def process_one_pair_files(pcd_file, label_file, calib_file):
  calibration = calibration_kitti.Calibration(calib_file)
  label_infos = process_label_file(label_file)
  np_pts = read_velodyne_bin(pcd_file)
  samples = extracted_points(np_pts, label_infos, calibration)
  
  return samples


def dump_pts_file(file_path, data):
  pts = []
  for i in range(data.shape[0]):
    pts.append([float(data[i][j]) for j in range(data.shape[1])])
  with open(file_path, 'w') as fp:
    json.dump(pts, fp, indent=2)


def main():
  pcd_files, label_files, calib_files = process_pcd_files("../dataset/training")
  ped_id = 0
  cyclist_id = 0
  veh_id = 0
  genobj_id = 0

  for i in tqdm(range(len(pcd_files)), desc='File Processing'):
  # for i in range(len(pcd_files)):
    samples = process_one_pair_files(pcd_files[i], label_files[i], calib_files[i])
    for sample in samples:
      if sample['class'] == 'Car':
        dump_pts_file('./dataset/car/' + str(veh_id) + '.json', sample['pts'])
        veh_id += 1
      elif sample['class'] == 'Cyclist':
        dump_pts_file('./dataset/cyclist/' + str(cyclist_id) + '.json', sample['pts'])
        cyclist_id += 1
      elif sample['class'] == 'Pedestrian':
        dump_pts_file('./dataset/ped/' + str(ped_id) + '.json', sample['pts'])
        ped_id += 1
      elif sample['class'] == 'genobj':
        dump_pts_file('./dataset/genobj/' + str(genobj_id) + '.json', sample['pts'])
        genobj_id += 1
    # break

if __name__ == "__main__":
  main()
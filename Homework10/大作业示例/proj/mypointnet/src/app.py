import open3d as o3d
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PointNetDataset
from model import PointNet



SEED = 13
gpus = [0]
batch_size = 1
ckp_path = '../output/latest.pth'


def load_ckp(ckp_path, model):
  state = torch.load(ckp_path)
  model.load_state_dict(state['state_dict'])
  print("model load from %s" % ckp_path)


def show_np_pts3d(np_pts):
  pc_view = o3d.geometry.PointCloud()
  pc_view.points = o3d.utility.Vector3dVector(np_pts)
  o3d.visualization.draw_geometries([pc_view])


if __name__ == "__main__":
  torch.manual_seed(SEED)
  device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
  print("Loading app dataset...")
  app_data = PointNetDataset("../../dataset", train=3)
  app_loader = DataLoader(app_data, batch_size=batch_size, shuffle=True)
  model = PointNet().to(device=device)
  if ckp_path:
    load_ckp(ckp_path, model)
    model = model.to(device)
  
  model.eval()

  with torch.no_grad():
    for x, y in app_loader:
      x = x.to(device)
      y = y.to(device)

      out = model(x)

      pred_y = np.argmax(out.cpu().numpy(), axis=1)
      gt = np.argmax(y.cpu().numpy(), axis=1)
      print("This pile of pointclouds is predicted to be[" + app_data.classes()[gt[0]] + "]")

      pts = x.cpu().numpy()[0].T
      show_np_pts3d(pts)
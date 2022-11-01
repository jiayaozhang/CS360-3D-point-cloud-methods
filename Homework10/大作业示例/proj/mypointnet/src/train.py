from dataset import PointNetDataset
from model import PointNet
from util import get_eval_acc_results

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch
import numpy as np
import time
import os
import datetime

import faulthandler

SEED = 13
batch_size = 32
epochs = 10000
decay_lr_factor = 0.95
decay_lr_every = 2
lr = 0.01
gpus = [0]
show_every = 1
val_every = 3
date = datetime.date.today()
save_dir = "../output"
global_step = 0


def save_ckp(ckp_dir, model, optimizer, epoch, best_acc, date):
  os.makedirs(ckp_dir, exist_ok=True)
  state = {
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
    }
  ckp_path = os.path.join(ckp_dir, f'date_{date}-epoch_{epoch}-maxacc_{best_acc:.3f}.pth')
  torch.save(state, ckp_path)
  torch.save(state, os.path.join(ckp_dir,f'latest.pth'))
  print('model saved to %s' % ckp_path)


def load_ckp(ckp_path, model, optimizer):
  state = torch.load(ckp_path)
  model.load_state_dict(state['state_dict'])
  optimizer.load_state_dict(state['optimizer'])
  print("model load from %s" % ckp_path)


def softXEnt(input, target):
  logprobs = torch.nn.functional.log_softmax(input, dim=1)
  return -(target * logprobs).sum() / input.shape[0]


if __name__ == "__main__":
  faulthandler.enable()
  torch.manual_seed(SEED)
  device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
  print("Loading train dataset...")
  train_data = PointNetDataset("../../dataset", train=0)
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  print("Loading valid dataset...")
  val_data = PointNetDataset("../../dataset", train=2)
  val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
  print("Set model and optimizer...")
  model = PointNet().to(device=device)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)
  writer = SummaryWriter('../output/runs/tersorboard')

  best_acc = 0.0
  model.train()
  print("Start trainning...")
  for epoch in range(epochs):
    acc_loss = 0.0
    num_samples = 512
    start_tic = time.time()
    for x, y in train_loader:
      x = x.to(device)
      y = y.to(device)
      optimizer.zero_grad()
      out = model(x)
      loss = softXEnt(out, y)
      # print('acc: ', acc)
      loss.backward()
      optimizer.step()
      acc = np.sum(np.argmax(out.cpu().detach().numpy(), axis=1) == 
          np.argmax(y.cpu().detach().numpy(), axis=1)) / len(y)
      acc_loss += batch_size * loss.item()
      num_samples += y.shape[0]
      global_step += 1
      if (global_step + 1) % show_every == 0:
        # ...log the running loss
        writer.add_scalar('training loss', acc_loss / num_samples, global_step)
        writer.add_scalar('training acc', acc, global_step)
        # print( f"loss at epoch {epoch} step {global_step}:{loss.item():3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
    scheduler.step()
    print(f"loss at epoch {epoch}:{acc_loss / num_samples:.3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
    
    if (epoch + 1) % val_every == 0:
      acc = get_eval_acc_results(model, val_loader, device)
      print("eval at epoch[" + str(epoch) + f"] acc[{acc:3f}]")
      writer.add_scalar('validing acc', acc, global_step)
      if acc > best_acc:
        best_acc = acc
        save_ckp(save_dir, model, optimizer, epoch, best_acc, date)

        example = torch.randn(1, 3, 10000).to(device)
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save("../output/traced_model.pt")
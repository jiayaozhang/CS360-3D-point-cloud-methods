import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
import os

import tensorboard_logger as tb_log
import argparse
import numpy as np
from pointnet import PointNet
from torch.utils.data import DataLoader
from dataloader import ModelNet40DataSet

k = 40
MLP = [[3,64,64],[64,64,128,1024]]

CLSMLP = [1024,512,256,k]

def getmodel(segmentation: bool = False):
    return PointNet(mlp1 = MLP[0],mlp2 = MLP[1],tailmlp=CLSMLP)

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--batch_size", type=int, default=40)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--ckpt_save_interval", type=int, default=1)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--ckpt", type=str, default='None')
parser.add_argument("--save_best", default=True)

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_decay', type=float, default=0.5)
parser.add_argument('--decay_step_list', type=list, default=[20,30,50, 70, 80, 90])

parser.add_argument('--weight_decay', type=float, default=1e-8)

parser.add_argument("--output_dir", type=str, default='./output')

args = parser.parse_args()

def log_print(info, log_f=None):
    print(info)
    if log_f is not None:
        print(info, file=log_f)


def save_checkpoint(model, epoch, ckpt_name):
    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    state = {'epoch': epoch, 'model_state': model_state}
    ckpt_name = '{}.pth'.format(ckpt_name)
    torch.save(state, ckpt_name)

def load_checkpoint(model, filename):
    print(filename)
    if os.path.isfile(filename):
        log_print("==> Loading from checkpoint %s" % filename)
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        log_print("==> Done")
    else:
        raise FileNotFoundError

    return epoch

def train_and_eval(model, train_data, eval_data, tb_log, ckpt_dir, log_f, last_epoch = -1):

    model.cuda() # 将模型迁移到GPU上
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #设定优化器

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = args.decay_step_list, gamma=args.lr_decay, last_epoch = last_epoch)

    total_it = 0
    best_acc = -1e+8
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):

        total_it = train_one_epoch(model, train_data, optimizer, epoch, lr_scheduler, total_it, tb_log, log_f)
        lr_scheduler.step(epoch)
        if epoch % args.ckpt_save_interval == 0:
            with torch.no_grad():
                avg_acc = eval_one_epoch(model, eval_data, epoch, tb_log = tb_log, log_f = log_f)
                if args.save_best and avg_acc > best_acc:
                    best_acc = avg_acc
                    best_epoch = epoch
                    ckpt_name = os.path.join(ckpt_dir, 'checkpoint_bestepoch')
                    save_checkpoint(model, epoch, ckpt_name)

    log_print('training end ,best epoch occur at %d and best avg acc is %.5f' %(best_epoch,best_acc), log_f=log_f)

def train_one_epoch(model, train_loader, optimizer, epoch, lr_scheduler, total_it, tb_log, log_f):
    model.train()
    log_print('===============TRAIN EPOCH %d================' % epoch, log_f=log_f)

    loss_func = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    train_loss = 0
    itnum = 0

    for it, batch in enumerate(train_loader):
        itnum += 1

        optimizer.zero_grad()

        pts_input, cls_labels = batch['pts_input'], batch['cls_labels']
        # print(pts_input.shape)
        pts_input = pts_input.cuda().float()
        # pts_input = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
        # cls_labels = torch.from_numpy(cls_labels).cuda(non_blocking=True).long().view(-1)
        cls_labels = cls_labels.cuda().long().view(-1)
       
        pred_cls = model(pts_input)
        pred_cls = pred_cls.squeeze()

        loss = loss_func(pred_cls, cls_labels)
        train_loss += loss.item()
        loss.backward()
        # clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_it += 1

        _,pred = torch.max(pred_cls,1)

        total += cls_labels.shape[0]

        correct += (pred == cls_labels).sum().item()

        cur_lr = lr_scheduler.get_lr()[0]
        tb_log.log_value('learning_rate', cur_lr, epoch)
        # if tb_log is not None:
        #     tb_log.log_value('trainit_loss', loss, total_it)
        #     tb_log.log_value('trainit_acc', acc, total_it)

        log_print('training epoch %d: it=%d/%d, total_it=%d, loss=%.5f, lr=%f' %
                  (epoch, it, len(train_loader), total_it, loss.item(), cur_lr), log_f=log_f)

    train_acc = correct / total
    train_loss = train_loss / itnum
    tb_log.log_value('train_acc', train_acc, epoch)
    tb_log.log_value('train_loss', train_loss, epoch)

    return total_it
    
def eval_one_epoch(model, eval_loader, epoch, tb_log=None, log_f=None):
    model.eval()
    log_print('===============EVAL EPOCH %d================' % epoch, log_f=log_f)

    correct = 0
    total = 0
    loss_func = nn.CrossEntropyLoss()
    itnum = 0
    loss = 0

    for it, batch in enumerate(eval_loader):

        pts_input, cls_labels = batch['pts_input'], batch['cls_labels']
        # pts_input = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
        pts_input = pts_input.cuda().float()
        # cls_labels = torch.from_numpy(cls_labels).cuda(non_blocking=True).long().view(-1)
        pred_cls = model(pts_input)
        pred_cls = pred_cls.squeeze()

        cls_labels = cls_labels.cuda().long().view(-1)
        loss += loss_func(pred_cls,cls_labels)
        itnum += 1
        
        _,pred_cls = torch.max(pred_cls,1)
        total += cls_labels.shape[0]
        correct += (pred_cls == cls_labels).sum().item()


    avg_acc = correct / total
    loss = loss / itnum


    tb_log.log_value('test_acc', avg_acc, epoch)
    tb_log.log_value('test_loss', loss, epoch)

    log_print('\nEpoch %d: Average acc: %.6f' % (epoch, avg_acc), log_f=log_f)
    return avg_acc



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    # output dir config
    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    tb_log.configure(os.path.join(output_dir, 'tensorboard'))
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    log_file = os.path.join(output_dir, 'log.txt')
    log_f = open(log_file, 'w')

    train_set = ModelNet40DataSet(traindata = True)
    eval_set = ModelNet40DataSet(traindata = False)

    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True,pin_memory=True,num_workers=args.workers)
    eval_loader = DataLoader(eval_set,batch_size=args.batch_size,shuffle=False,pin_memory=True,num_workers=args.workers)

    epoch = -1
    model = getmodel()

    if args.mode == 'train':
        if args.ckpt != 'None':
            epoch = load_checkpoint(model,args.ckpt)
        if torch.cuda.device_count() > 1:
            print("let's use ",torch.cuda.device_count(),"GPUs")
            model = nn.DataParallel(model)

        for key, val in vars(args).items():
            log_print("{:16} {}".format(key, val), log_f=log_f)
        # train and eval
        train_and_eval(model, train_loader, eval_loader, tb_log, ckpt_dir, log_f,last_epoch = epoch)
        log_f.close()

    elif args.mode == 'eval':
        if not isinstance(model, torch.nn.DataParallel):
            args.batch_size = 2
        epoch = load_checkpoint(model, args.ckpt)
        model.cuda()
        with torch.no_grad():
            avg_iou = eval_one_epoch(model, eval_loader, epoch, log_f=log_f)
    else:
        raise NotImplementedError


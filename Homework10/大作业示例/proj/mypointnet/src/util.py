import torch
from typing import List
import numpy as np
from sklearn.metrics import classification_report

def get_eval_acc_results(model, data_loader, device):
    """
    ACC
    """
    seq_id = 0
    model.eval()

    distribution = np.zeros([5])
    confusion_matrix = np.zeros([5, 5])
    pred_ys = []
    gt_ys = []
    with torch.no_grad():
        accs = []
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            # out = model(np.array([i.handcraft for i in data]))

            pred_y = np.argmax(out.cpu().numpy(), axis=1)
            gt = np.argmax(y.cpu().numpy(), axis=1)

            acc = np.sum(pred_y == gt) / len(pred_y)
            gt_ys = np.append(gt_ys, gt)
            pred_ys = np.append(pred_ys, pred_y)
            idx = gt

            accs.append(acc)

        return np.mean(accs)
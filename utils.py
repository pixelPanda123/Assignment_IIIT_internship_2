import os
import json
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_plot(train_losses, val_losses, out_path, title):
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def save_json(obj, out_path):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, 'w') as f:
        json.dump(obj, f, indent=2)

def perplexity(loss):
    return float(torch.exp(torch.tensor(loss)))
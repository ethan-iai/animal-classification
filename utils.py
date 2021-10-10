import os
import base64
import shutil
import logging
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from urllib.request import urlretrieve

import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

from PIL import Image

def create_onedrive_directdownload(onedrive_link):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    result_url = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return result_url

def download_rsc(url, filename, show_progress_bar=True):
    '''
    Download resource at ```url``` and save it to ```path```  
    '''
    hook = None if not show_progress_bar else _download_rsc_tqdm_hook(tqdm(unit='B', unit_scale=True))
    urlretrieve(url, filename, hook)

def _download_rsc_tqdm_hook(pbar):
    '''Wrapper for tqdm'''
    downloaded = [0]

    def update(count, block_size, total_size):
        if total_size is not None:
            pbar.total = total_size
        delta = count * block_size - downloaded[0]
        downloaded[0] = count * block_size
        pbar.update()

    return update

def read_image(path, shape=None, resample=None):
    """ read single image

    Args:
        path ([type]): [description]
        shape ([type], optional): [description]. Defaults to None.
        resample ([type], optional): [description]. Defaults to None.

    Returns:
        [PIL.Image]: [description]
    """
    try:
        return Image.open(path).convert("RGB")
    except:
        return None
    # if shape is not None:
    #     img = img.resize(shape, resample)
    # return np.array(img)

def read_images(path, shape=None, resample=None):
    """ read images from directory path

    Args:
        path ([type]): [description]
        dtype ([type]): [description]
        shape ([type], optional): [description]. Defaults to None.
        resample ([type], optional): [description]. Defaults to None.

    Returns:
        imgs (list[PIL.Image]): [description]
    """

    imgs = []
    valid_exts = \
        set([".jpg", ".gif", ".png", 
            ".tga", ".jpeg", ".ppm"])
    for img_name in sorted(os.listdir(path)):
        ext = os.path.splitext(img_name)[-1]
        if ext.lower() not in valid_exts:
            continue
        img = read_image(os.path.join(path, img_name), 
                         shape, resample)
        if img == None:
            continue
        imgs.append(img)
    # imgs = np.stack(imgs, axis=0)

    return imgs

def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def create_loss_fn(args):
    if args.label_smoothing > 0:
        criterion = SmoothCrossEntropy(alpha=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion.to(args.device)


def module_load_state_dict(model, state_dict):
    try:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = f'module.{k}'  # add `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def model_load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        module_load_state_dict(model, state_dict)


def save_checkpoint(args, state, is_best, finetune=False):
    os.makedirs(args.save_path, exist_ok=True)
    if finetune:
        name = f'{args.name}_finetune'
    else:
        name = args.name
    filename = f'{args.save_path}/{name}_last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    if is_best:
        shutil.copyfile(filename, f'{args.save_path}/{args.name}_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    output = output.to(torch.device('cpu'))
    target = target.to(torch.device('cpu'))
    maxk = max(topk)
    batch_size = target.shape[0]

    _, idx = output.sort(dim=1, descending=True)
    pred = idx.narrow(1, 0, maxk).t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

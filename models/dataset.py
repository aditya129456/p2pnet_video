import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as standard_transforms


class DeNormalize(object):
    """Denormalize a tensor image using mean and standard deviation."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class CrowdDataset(Dataset):
    """
    Standardized Crowd Counting Dataset loader.
    Supports ShanghaiTech (Part A/B), Mall Dataset, JHU-CROWD++, and custom datasets
    formatted with train_list.txt / val_list.txt pointing to image and coordinate files.
    """
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root

        train_list_candidates = ['train_list.txt', 'list.txt']
        eval_list_candidates = ['val_list.txt', 'tlist.txt', 'test_list.txt']

        self.train_lists = ""
        for name in train_list_candidates:
            p = os.path.join(data_root, name)
            if os.path.exists(p):
                with open(p, 'r') as f:
                    self.train_lists = f.read()
                break

        self.eval_list = ""
        for name in eval_list_candidates:
            p = os.path.join(data_root, name)
            if os.path.exists(p):
                with open(p, 'r') as f:
                    self.eval_list = f.read()
                break

        if train:
            self.img_list_file = [l.strip() for l in self.train_lists.split('\n') if l.strip()]
        else:
            self.img_list_file = [l.strip() for l in self.eval_list.split('\n') if l.strip()]

        self.img_map = {}
        for train_list in self.img_list_file:
            parts = train_list.split(' ')
            if len(parts) == 2:
                img_rel = parts[0].lstrip('/\\')
                gt_rel = parts[1].lstrip('/\\')
                self.img_map[os.path.join(self.root_path, img_rel)] = \
                    os.path.join(self.root_path, gt_rel)

        self.img_list = sorted(list(self.img_map.keys()))
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]

        img, point = load_data((img_path, gt_path), self.train)

        # align validation images to multiples of 128 for FPN multi-scale feature consistency
        # and cap max dimension at 1536 to prevent CUDA OOM on ultra-high-resolution images
        if not self.train:
            w, h = img.size
            max_dim = 1536
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                target_w = int(w * scale)
                target_h = int(h * scale)
            else:
                target_w, target_h = w, h
            new_w = max(128, (target_w // 128) * 128)
            new_h = max(128, (target_h // 128) * 128)
            if new_w != w or new_h != h:
                scale_x = new_w / w
                scale_y = new_h / h
                resample_mode = Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR
                img = img.resize((new_w, new_h), resample_mode)
                if len(point) > 0:
                    point = point.copy()
                    point[:, 0] *= scale_x
                    point[:, 1] *= scale_y

        if self.transform is not None:
            img = self.transform(img)
        elif not isinstance(img, torch.Tensor):
            import torchvision.transforms.functional as TF
            img = TF.to_tensor(img)

        if self.train:
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            if scale * min_size > 128:
                img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)
                point *= scale

        if self.train and self.patch:
            img, point = random_crop(img, point)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
        else:
            point = [torch.as_tensor(point, dtype=torch.float32).reshape(-1, 2)]

        if random.random() > 0.5 and self.train and self.flip:
            if isinstance(img, torch.Tensor):
                img = torch.flip(img, dims=[-1])
            else:
                img = torch.Tensor(img[..., ::-1].copy())
            w_dim = 128 if (self.train and self.patch) else img.shape[-1]
            for i, _ in enumerate(point):
                if len(point[i]) > 0:
                    point[i][:, 0] = w_dim - point[i][:, 0]

        img = torch.Tensor(img)
        target = [{} for _ in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.as_tensor(point[i], dtype=torch.float32).reshape(-1, 2)
            fname = os.path.basename(img_path)
            base = os.path.splitext(fname)[0]
            digits = ''.join(filter(str.isdigit, base))
            image_id = int(digits) if digits else (index + 1)
            target[i]['image_id'] = torch.Tensor([image_id]).long()
            target[i]['labels'] = torch.ones([target[i]['point'].shape[0]]).long()

        return img, target


# Alias for backwards compatibility
SHHA = CrowdDataset


def load_data(img_gt_path, train):
    """Load image from path and corresponding coordinates from GT txt file."""
    img_path, gt_path = img_gt_path
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at: {img_path}")
    img = Image.open(img_path).convert('RGB')
    points = []

    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f_label:
            for line in f_label:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    points.append([x, y])

    return img, np.array(points, dtype=np.float32).reshape(-1, 2)


def random_crop(img, den, num_patch=4):
    """Random patch cropping for scale and translation invariant training."""
    half_h = 128
    half_w = 128

    pad_h = max(0, half_h - img.size(1))
    pad_w = max(0, half_w - img.size(2))
    if pad_h > 0 or pad_w > 0:
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)

    max_h = max(0, img.size(1) - half_h)
    max_w = max(0, img.size(2) - half_w)

    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w], dtype=np.float32)
    result_den = []
    den = np.asarray(den, dtype=np.float32).reshape(-1, 2)

    for i in range(num_patch):
        start_h = random.randint(0, max_h) if max_h > 0 else 0
        start_w = random.randint(0, max_w) if max_w > 0 else 0
        end_h = start_h + half_h
        end_w = start_w + half_w
        result_img[i] = img[:, start_h:end_h, start_w:end_w]

        if len(den) > 0:
            idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
            record_den = den[idx].copy()
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h
        else:
            record_den = np.zeros((0, 2), dtype=np.float32)

        result_den.append(record_den)

    return result_img, result_den


def loading_data(data_root):
    """Build train and validation CrowdDataset instances with standard normalization."""
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.532, 0.46, 0.41],
                                      std=[0.232, 0.215, 0.202]),
    ])
    train_set = CrowdDataset(data_root, train=True, transform=transform, patch=True, flip=True)
    val_set = CrowdDataset(data_root, train=False, transform=transform)

    return train_set, val_set


def build_dataset(args=None):
    """Dataset factory entrypoint matching standard P2PNet training pipeline."""
    return loading_data

import numpy as np
import argparse
import os
import cv2
import random
from tqdm import tqdm
# import uuid
import gc
import psutil
# import shutil
import datetime
import json
from PIL import Image
from scipy.interpolate import Rbf
from collections import Counter
import seaborn as sns
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from skimage import io
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import multiprocessing

config = None

# Parameter configuration
class Config:
    # ======================================================================================================================
    def __init__(self, **kwargs):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.save_model_prefix = 'HPLmodel'
        self.save_root_dir = "./samezoom_multilabel090404"
        self.image_dir = './same_zoom_selected250812_rename'
        self.class_names = []
        self.class_names_cn = []
        self.label_file = None
        self.annotation_file = None

        self.num_classes = None
        self.sample_num = 359
        self.target_size = (224, 224)
        self.num_epochs = 150
        self.batch_size = 32
        self.test_size = 0.2
        self.val_size = 0.2
        self.learning_rate = 5e-4
        self.weight_decay = 1e-4
        self.early_stop_patience = 15
        self.dropout_rate = 0.4
        self.hidden_size = 512

        self.random_seed = 40
        self.random_seeds = [42, 123]

        self.use_augmentation = True
        self.intensity_modifier = 3.0
        self.augment_counts_expand = 3
        self.augment_counts = {}
        self.num_samples_to_save = 5
        self.debug_mode = True
        self.use_binary_channel = True
        self.binary_image_suffix = '_binary.png'

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Config does not support parameter {key}")

        self.data_split_file = os.path.join(self.save_root_dir, "data_split.txt")
        self.augmentation_sample_dir = os.path.join(self.save_root_dir, "augmentation_samples")
        self.log_file = os.path.join(self.save_root_dir, "training_log.json")

        if not hasattr(self, 'augment_counts') or not self.augment_counts:
            if hasattr(self, 'class_names') and self.class_names:
                self.augment_counts = {name: 1 for name in self.class_names}
                self.augment_counts['default'] = 1
            else:
                self.augment_counts = {'default': 1}

        if self.augment_counts_expand > 1:
            for k in self.augment_counts:
                self.augment_counts[k] *= self.augment_counts_expand

        if self.label_file is None:
            print("Warning: label_file is not set; it will be passed in via the GUI.")
        if self.num_classes is None:
            print("Warning: num_classes is not set; it will be passed in via the GUI.")

        print(f"Config initialized successfully:")
        print(f"  - num_classes: {self.num_classes}")
        print(f"  - class_names: {self.class_names}")
        print(f"  - class_names_cn: {self.class_names_cn}")
        print(f"  - image_dir: {self.image_dir}")
        print(f"  - label_file: {self.label_file}")
        print(f"  - save_root_dir: {self.save_root_dir}")
        print(f"  - augment_counts: {self.augment_counts}")

    def save_config(self):
        config_dict = {key: value for key, value in vars(self).items() if not key.startswith('_')}
        config_dict['device'] = str(config_dict['device'])
        config_dict['target_size'] = list(config_dict['target_size'])
        return config_dict

def get_defect_mapping(config):
    if hasattr(config, 'class_names') and config.class_names:
        defect_types = {i: name for i, name in enumerate(config.class_names)}
        defect_names_cn = {i: name for i, name in enumerate(getattr(config, 'class_names_cn', config.class_names))}
    else:
        defect_types = {0: 'no_defect', 1: 'dislocation', 2: 'bridges', 3: 'junction'}
        defect_names_cn = {0: '无缺陷', 1: '位错', 2: '桥接', 3: '交点'}
    return defect_types, defect_names_cn

def binarize_image(image):
    """Perform binarization on the input image and return a PIL image."""
    image = np.array(image)
    for _ in range(2):
        image = cv2.medianBlur(image, 9)
    binary_image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 205, 1
    )
    for _ in range(5):
        binary_image = cv2.medianBlur(binary_image, 9)
    return Image.fromarray(binary_image)

def print_dataset_distribution(name, image_paths, labels):
    """Print the sample distribution of the dataset."""
    print(f"\n{name} Sample distribution of the dataset：")
    print(f"Sample number：{len(image_paths)}")
    class_counts = [np.sum(labels[:, i]) for i in range(config.num_classes)]
    for i, count in enumerate(class_counts):
        defect_types, defect_names_cn = get_defect_mapping(config)
        print(f"  {defect_names_cn[i]}: {int(count)} positive samples")
    return class_counts

def save_data_split(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, seed=None):
    """Save dataset split to JSON and CSV files."""
    defect_types, defect_names_cn = get_defect_mapping(config)
    os.makedirs(os.path.join(config.save_root_dir, 'data_splits'), exist_ok=True)

    split_data = {
        "seed": seed,
        "test_set": {
            "image_paths": test_paths,
            "labels": test_labels.tolist(),
            "class_distribution": [int(np.sum(test_labels[:, i])) for i in range(config.num_classes)]
        }
    }
    if seed is not None:
        split_data["train_set"] = {
            "image_paths": train_paths,
            "labels": train_labels.tolist(),
            "class_distribution": [int(np.sum(train_labels[:, i])) for i in range(config.num_classes)]
        }
        split_data["val_set"] = {
            "image_paths": val_paths,
            "labels": val_labels.tolist(),
            "class_distribution": [int(np.sum(val_labels[:, i])) for i in range(config.num_classes)]
        }

    json_path = os.path.join(config.save_root_dir, 'data_splits',
                             f'data_split{"_seed" + str(seed) if seed else ""}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(split_data, f, indent=4, ensure_ascii=False)
    if config.debug_mode:
        print(f"Save data split to {json_path}.")

    if seed is None:
        distribution_data = {
            "Dataset": ["Test"],
            **{defect_names_cn[i]: [int(np.sum(test_labels[:, i]))] for i in range(config.num_classes)}
        }
    else:
        distribution_data = {
            "Dataset": ["Train", "Validation", "Test"],
            **{defect_names_cn[i]: [
                int(np.sum(train_labels[:, i])),
                int(np.sum(val_labels[:, i])),
                int(np.sum(test_labels[:, i]))
            ] for i in range(config.num_classes)}
        }

    df = pd.DataFrame(distribution_data)
    csv_path = os.path.join(config.save_root_dir, 'data_splits',
                            f'class_distribution{"_seed" + str(seed) if seed else ""}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    if config.debug_mode:
        print(f"Save data split to {csv_path}.")

    mode = 'a' if seed is not None else 'w'
    with open(config.data_split_file, mode, encoding='utf-8') as f:
        if seed is not None:
            f.write(f"\n=== Seed {seed} ===\n")
        else:
            f.write("=== Dataset Split ===\n")
        f.write("\nTest set:\n")
        f.write(f"Number of samples: {len(test_paths)}\n")
        f.write("Image path | Label\n")
        for path, label in zip(test_paths, test_labels):
            defect_names = [defect_names_cn[i] for i in range(config.num_classes) if label[i] == 1]
            f.write(f"{path} | {', '.join(defect_names) if defect_names else 'No defect'}\n")
        if seed is not None:
            f.write("\nTraining set:\n")
            f.write(f"Number of samples: {len(train_paths)}\n")
            f.write("Image path | Label\n")
            for path, label in zip(train_paths, train_labels):
                defect_names = [defect_names_cn[i] for i in range(config.num_classes) if label[i] == 1]
                f.write(f"{path} | {', '.join(defect_names) if defect_names else 'No defect'}\n")
            f.write("\nValidation set:\n")
            f.write(f"Number of samples: {len(val_paths)}\n")
            f.write("Image path | Label\n")
            for path, label in zip(val_paths, val_labels):
                defect_names = [defect_names_cn[i] for i in range(config.num_classes) if label[i] == 1]
                f.write(f"{path} | {', '.join(defect_names) if defect_names else 'No defect'}\n")
    if config.debug_mode:
        print(f"Data split saved to {config.data_split_file}")

class AdvancedImageDataset(Dataset):
    def __init__(self, image_paths, labels, annotations, augment=False, model=None, use_binary_channel=False, config=None):
        self.image_paths = image_paths
        self.labels = labels
        self.annotations = annotations
        self.augment = augment
        self.model = model
        self.use_binary_channel = use_binary_channel
        self.config = config
        self.label_inconsistencies = []
        self.image_cache = {}
        self.binary_cache = {}
        if self.config.debug_mode:
            print(
                f"Initializing dataset: {len(image_paths)} samples, augment={augment}, use_binary_channel={use_binary_channel}")

        if self.config.debug_mode:
            print(f"Preloading {len(image_paths)} images into memory")
        max_cache_size = 1024
        process = psutil.Process()
        available_memory = psutil.virtual_memory().available / 1024 / 1024
        if available_memory < max_cache_size:
            if self.config.debug_mode:
                print(f"Available memory {available_memory:.2f} MB is insufficient, disabling image caching")
            self.image_cache = None
            self.binary_cache = None
        else:
            for idx, path in enumerate(image_paths):
                img = io.imread(path, as_gray=True)
                self.image_cache[idx] = img
                if use_binary_channel:
                    img = Image.fromarray(img).resize((640, 480), Image.Resampling.LANCZOS)
                    self.binary_cache[idx] = binarize_image(img)

        self.base_transform = transforms.Compose([
            transforms.Resize(self.config.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, scale=(0.85, 0.9)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])

        defect_types, defect_names_cn = get_defect_mapping(self.config)
        self.augmentation_per_image = []
        self.cumulative_samples = [0]
        total_samples = 0
        for idx in range(len(image_paths)):
            label = labels[idx]
            defect_names = [defect_types[i] for i in range(self.config.num_classes) if label[i] == 1]
            aug_count = max(self.config.augment_counts.get(dname, self.config.augment_counts['default'])
                            for dname in defect_names) if defect_names else self.config.augment_counts['default']
            if not augment:
                aug_count = 0
            self.augmentation_per_image.append(aug_count)
            total_samples += (aug_count + 1)
            self.cumulative_samples.append(total_samples)

        self.augmentation_stats = Counter()
        for idx, label in enumerate(labels):
            aug_count = self.augmentation_per_image[idx]
            for i in range(self.config.num_classes):
                if label[i] == 1:
                    self.augmentation_stats[defect_types[i]] += (aug_count + 1)
        if self.config.debug_mode:
            print(f"Total samples after augmentation: {total_samples}")
            print(f"Augmentation distribution: {dict(self.augmentation_stats)}")

        if augment:
            self.save_augmentation_samples()

    # ====================================================下面是数据增强方法==================================================================

    def get_bbox_area(self, bbox, W, H):
        x_min, y_min, x_max, y_max = bbox
        x_min, x_max = max(0, x_min), min(W - 1, x_max)
        y_min, y_max = max(0, y_min), min(H - 1, y_max)
        return (x_max - x_min) * (y_max - y_min)

    def elastic_deform(self, img, defect_types_present=None, bboxes=None, is_binary=False):
        """Elastic deformation augmentation: minimum distance between control points, fixed absolute displacement range, with random sign."""
        img = np.array(img)
        H, W = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]
        rng = np.random.default_rng()
        intensity_modifier = getattr(self.config, 'intensity_modifier', 3.0)
        min_dist_ratio = 0.12
        min_dist = int(min(H, W) * min_dist_ratio)
        max_attempts = 100
        global_intensity_scale = 0.6
        border_pts = np.concatenate([
            np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]]),
            np.array([[x, 0] for x in np.linspace(0, W-1, 5)[1:-1]]),
            np.array([[x, H-1] for x in np.linspace(0, W-1, 5)[1:-1]]),
            np.array([[0, y] for y in np.linspace(0, H-1, 5)[1:-1]]),
            np.array([[W-1, y] for y in np.linspace(0, H-1, 5)[1:-1]])
        ])
        defect_params = {
            'no_defect': {'base_intensity': 0.03, 'base_points': 5},
            'dislocation': {'base_intensity': 0.01, 'base_points': 3},
            'bridges': {'base_intensity': 0.02, 'base_points': 3},
            'junction': {'base_intensity': 0.02, 'base_points': 3}
        }
        ctrl_pts = []
        dx_all = []
        dy_all = []
        all_points = []
        if bboxes:
            base_area = 5000
            for defect_type, bbox in bboxes:
                params = defect_params.get(defect_type, {'base_intensity': 0.02, 'base_points': 3})
                area = self.get_bbox_area(bbox, W, H)
                area_ratio = area / base_area if area > 0 else 1.0
                n_points = min(3, max(2, int(params['base_points'] + np.log1p(area_ratio))))
                intensity = params['base_intensity'] * global_intensity_scale * intensity_modifier
                x_min, y_min, x_max, y_max = bbox
                x_min, x_max = max(0, x_min), min(W - 1, x_max)
                y_min, y_max = max(0, y_min), min(H - 1, y_max)
                points = []
                for _ in range(n_points):
                    for attempt in range(max_attempts):
                        x = rng.uniform(x_min, x_max)
                        y = rng.uniform(y_min, y_max)
                        new_point = np.array([x, y])
                        if len(all_points) == 0 or np.all(np.sqrt(np.sum((np.array(all_points) - new_point)**2, axis=1)) >= min_dist):
                            points.append([x, y])
                            all_points.append([x, y])
                            break
                    else:
                        print(f"Warning: Unable to generate enough control points within bbox ({x_min}, {y_min}, {x_max}, {y_max})")
                if points:
                    points = np.array(points)
                    x0, y0 = points[:, 0], points[:, 1]
                    abs_values = rng.uniform(intensity * W * 0.3, intensity * W * 0.7, len(points))
                    signs = rng.choice([-1, 1], len(points))
                    dx = abs_values * signs
                    abs_values = rng.uniform(intensity * H * 0.3, intensity * H * 0.7, len(points))
                    signs = rng.choice([-1, 1], len(points))
                    dy = abs_values * signs
                    ctrl_pts.append(np.column_stack([x0, y0]))
                    dx_all.extend(dx)
                    dy_all.extend(dy)
        else:
            for defect_type in (defect_types_present or ['default']):
                params = defect_params.get(defect_type, {'base_intensity': 0.02, 'base_points': 3})
                n_points = params['base_points']
                intensity = params['base_intensity'] * global_intensity_scale * intensity_modifier
                points = []
                for _ in range(n_points):
                    for attempt in range(max_attempts):
                        x = rng.uniform(0.1 * W, 0.9 * W)
                        y = rng.uniform(0.1 * H, 0.9 * H)
                        new_point = np.array([x, y])
                        if len(all_points) == 0 or np.all(np.sqrt(np.sum((np.array(all_points) - new_point)**2, axis=1)) >= min_dist):
                            points.append([x, y])
                            all_points.append([x, y])
                            break
                    else:
                        print(f"Warning: Unable to generate enough control points in the image ({W}, {H})")
                if points:
                    points = np.array(points)
                    x0, y0 = points[:, 0], points[:, 1]
                    abs_values = rng.uniform(intensity * W * 0.3, intensity * W * 1.0, len(points))
                    signs = rng.choice([-1, 1], len(points))
                    dx = abs_values * signs
                    abs_values = rng.uniform(intensity * H * 0.3, intensity * H * 1.0, len(points))
                    signs = rng.choice([-1, 1], len(points))
                    dy = abs_values * signs
                    ctrl_pts.append(np.column_stack([x0, y0]))
                    dx_all.extend(dx)
                    dy_all.extend(dy)
        if len(ctrl_pts) > 0:
            ctrl_pts = np.vstack([border_pts] + ctrl_pts)
            dx_all = np.hstack([np.zeros(len(border_pts)), dx_all])
            dy_all = np.hstack([np.zeros(len(border_pts)), dy_all])
            max_points = 24
            if len(ctrl_pts) > max_points:
                indices = np.random.choice(len(ctrl_pts) - len(border_pts), max_points - len(border_pts), replace=False)
                ctrl_pts = np.vstack([border_pts, ctrl_pts[len(border_pts):][indices]])
                dx_all = np.hstack([np.zeros(len(border_pts)), dx_all[len(border_pts):][indices]])
                dy_all = np.hstack([np.zeros(len(border_pts)), dy_all[len(border_pts):][indices]])
        else:
            ctrl_pts = border_pts
            dx_all = np.zeros(len(border_pts))
            dy_all = np.zeros(len(border_pts))
        rbf_dx = Rbf(ctrl_pts[:, 0], ctrl_pts[:, 1], dx_all, function='thin_plate')
        rbf_dy = Rbf(ctrl_pts[:, 0], ctrl_pts[:, 1], dy_all, function='thin_plate')
        rows, cols = np.indices((H, W))
        coords = np.column_stack([cols.ravel(), rows.ravel()])
        delta_x = rbf_dx(coords[:, 0], coords[:, 1]).reshape(H, W)
        delta_y = rbf_dy(coords[:, 0], coords[:, 1]).reshape(H, W)
        map_x = (cols + delta_x).astype(np.float32)
        map_y = (rows + delta_y).astype(np.float32)
        border_value = 0 if channels == 1 else (0, 0, 0)
        deformed = cv2.remap(img, map_x, map_y, cv2.INTER_NEAREST if is_binary else cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
        if is_binary:
            deformed = (deformed > 127).astype(np.uint8) * 255
        return Image.fromarray(deformed), len(dx_all) - len(border_pts), ctrl_pts, dx_all, dy_all, border_pts

    def save_augmentation_samples(self):
        defect_types, defect_names_cn = get_defect_mapping(self.config)
        os.makedirs(self.config.augmentation_sample_dir, exist_ok=True)

        defect_indices = {defect_type: [] for defect_type in defect_types.values()}
        for idx in range(len(self.image_paths)):
            label = self.labels[idx]
            defect_names = [defect_types[i] for i in range(self.config.num_classes) if label[i] == 1]
            for defect in defect_names:
                if defect in defect_indices:
                    defect_indices[defect].append(idx)
        sample_indices = []
        used_indices = set()
        for defect in defect_names:
            available_indices = [idx for idx in defect_indices[defect] if idx not in used_indices]
            selected_indices = random.sample(available_indices, min(self.config.num_samples_to_save, len(available_indices)))
            sample_indices.extend(selected_indices)
            used_indices.update(selected_indices)

        for idx in sample_indices:
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            defect_names = [defect_types[i] for i in range(self.config.num_classes) if label[i] == 1]
            aug_count = self.augmentation_per_image[idx]
            img = io.imread(img_path, as_gray=True)
            W_orig, H_orig = img.shape[1], img.shape[0]
            img = Image.fromarray(img).resize((640, 480), Image.Resampling.LANCZOS)
            bboxes = [(bbox['defect_type'], [x * 640 / W_orig, y * 480 / H_orig, x2 * 640 / W_orig, y2 * 480 / H_orig])
                      for bbox in self.annotations.get(os.path.basename(img_path).split('.')[0], [])
                      for x, y, x2, y2 in [bbox['bbox']]]

            base_name = os.path.basename(img_path).split('.')[0]

            for aug_idx in range(min(5, aug_count)):
                elastic_img, n_ctrl_points, ctrl_pts, dx_all, dy_all, border_pts = self.elastic_deform(
                    img, defect_types_present=defect_names, bboxes=bboxes)
                elastic_img_binary = binarize_image(elastic_img)
                aug_seed = random.randint(0, 2 ** 32)
                torch.manual_seed(aug_seed)
                aug_img = self.augment_transform(elastic_img)
                torch.manual_seed(aug_seed)
                aug_img_binary = self.augment_transform(elastic_img_binary)

                if self.config.debug_mode:
                    plt.figure(figsize=(18, 5))
                    plt.subplot(1, 4 if self.use_binary_channel else 3, 1)
                    plt.title(f"original image\nlabel: {', '.join(defect_names)}")
                    plt.imshow(np.array(img), cmap='gray')
                    for defect_type, bbox in bboxes:
                        x_min, y_min, x_max, y_max = bbox
                        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                            fill=False, edgecolor='red', linewidth=2)
                        plt.gca().add_patch(rect)
                        plt.text(x_min, y_min - 10, f"{defect_type}", color='red')
                    plt.axis('off')

                    plt.subplot(1, 4 if self.use_binary_channel else 3, 2)
                    plt.title(f"elastic deformed image {aug_idx + 1}")
                    plt.imshow(np.array(elastic_img), cmap='gray')
                    for defect_type, bbox in bboxes:
                        x_min, y_min, x_max, y_max = bbox
                        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                            fill=False, edgecolor='red', linewidth=2)
                        plt.gca().add_patch(rect)
                        plt.text(x_min, y_min - 10, f"{defect_type}", color='red')
                    non_border_pts = ctrl_pts[len(border_pts):]
                    non_border_dx = dx_all[len(border_pts):]
                    non_border_dy = dy_all[len(border_pts):]
                    if len(non_border_pts) > 0:
                        plt.scatter(non_border_pts[:, 0], non_border_pts[:, 1], c='red', s=50, alpha=0.7,
                                    label='Control Points')
                        plt.quiver(non_border_pts[:, 0], non_border_pts[:, 1],
                                   non_border_dx, non_border_dy, color='green', scale=1, scale_units='xy',
                                   width=0.005, alpha=0.7, label='Movement')
                        plt.legend()
                    plt.axis('off')

                    plt.subplot(1, 4 if self.use_binary_channel else 3, 3)
                    plt.title(f"augmented image {aug_idx + 1}")
                    plt.imshow(np.array(aug_img), cmap='gray')
                    plt.axis('off')

                    if self.use_binary_channel:
                        plt.subplot(1, 4, 4)
                        plt.title(f"binary image {aug_idx + 1}")
                        plt.imshow(np.array(aug_img_binary), cmap='gray')
                        plt.axis('off')

                    save_path = os.path.join(self.config.augmentation_sample_dir,
                                             f"sample_{base_name}_augment_{aug_idx + 1}.png")
                    plt.savefig(save_path, bbox_inches='tight')
                    plt.close()
                    print(f"Save augmented samples to {save_path}.")

    def __len__(self):
        return self.cumulative_samples[-1]

    def __getitem__(self, idx):
        orig_idx = None
        for i in range(len(self.cumulative_samples) - 1):
            if self.cumulative_samples[i] <= idx < self.cumulative_samples[i + 1]:
                orig_idx = i
                break
        if orig_idx is None:
            raise IndexError(f"Index {idx} is out of dataset range; dataset size is {self.__len__()}.")

        aug_idx = idx - self.cumulative_samples[orig_idx]
        aug_count = self.augmentation_per_image[orig_idx]

        img_path = self.image_paths[orig_idx]
        label = self.labels[orig_idx]
        defect_types, defect_names_cn = get_defect_mapping(self.config)
        defect_names = [defect_types[i] for i in range(self.config.num_classes) if label[i] == 1]
        if self.image_cache is None:
            img = io.imread(img_path, as_gray=True)
        else:
            img = self.image_cache[orig_idx].copy()
        W_orig, H_orig = img.shape[1], img.shape[0]
        img = Image.fromarray(img).resize((640, 480), Image.Resampling.LANCZOS)
        binary_img = binarize_image(img)

        bboxes = [(bbox['defect_type'], [x * 640 / W_orig, y * 480 / H_orig, x2 * 640 / W_orig, y2 * 480 / H_orig])
                  for bbox in self.annotations.get(os.path.basename(img_path).split('.')[0], [])
                  for x, y, x2, y2 in [bbox['bbox']]]

        if aug_idx == 0:
            if self.augment:
                if self.use_binary_channel:
                    binary_img = binarize_image(img)
                aug_seed = random.randint(0, 2 ** 32)
                torch.manual_seed(aug_seed)
                img = self.augment_transform(img)
                if self.use_binary_channel:
                    torch.manual_seed(aug_seed)
                    binary_img = self.augment_transform(binary_img)
        elif aug_idx <= aug_count:
            if self.augment:
                img, _, _, _, _, _ = self.elastic_deform(img, defect_types_present=defect_names, bboxes=bboxes)
                if self.use_binary_channel:
                    binary_img = binarize_image(img)
                aug_seed = random.randint(0, 2 ** 32)
                torch.manual_seed(aug_seed)
                img = self.augment_transform(img)
                if self.use_binary_channel:
                    torch.manual_seed(aug_seed)
                    binary_img = self.augment_transform(binary_img)

        else:
            raise IndexError(f"Augmentation index {aug_idx} exceeds {aug_count}, image: {img_path}")

        img_tensor = self.base_transform(img)
        if self.use_binary_channel:
            binary_tensor = self.base_transform(binary_img)
            img_tensor = torch.cat([img_tensor, binary_tensor], dim=0)
        return img_tensor, torch.tensor(label, dtype=torch.float32)


# Focal Loss
class MultiLabelFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5):
        super(MultiLabelFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# =================================EfficientNet-B0+FPN+CBAM======================================================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.ca(x) * x
        max_out = torch.max(ca, dim=1, keepdim=True)[0]
        mean_out = torch.mean(ca, dim=1, keepdim=True)
        sa = self.sa(torch.cat([max_out, mean_out], dim=1)) * ca
        return sa

class SingleStreamEfficientCBAM(nn.Module):
    def __init__(self, num_classes=5, hidden_size=512, dropout_rate=0.5):
        super(SingleStreamEfficientCBAM, self).__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone.features[0][0] = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, bias=False)

        self.feature_extractor = create_feature_extractor(
            self.backbone,
            return_nodes={
                'features.2': 'stage1',  # Number of channels 24, Size 56x56
                'features.4': 'stage2',  # Number of channels 80, Size 14x14
                'features.6': 'stage3',  # Number of channels 192, Size 7x7
                'features.8': 'stage4'   # Number of channels 1280, Size 7x7
            }
        )

        self.fpn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(24, 256, kernel_size=1, groups=1),
                nn.MaxPool2d(kernel_size=2, stride=2)  # 56x56 -> 28x28
            ),  # stage1
            nn.Conv2d(80, 256, kernel_size=1, groups=1),  # stage2
            nn.Conv2d(192, 256, kernel_size=1, groups=1), # stage3
            nn.Conv2d(1280, 256, kernel_size=1, groups=1) # stage4
        ])
        self.cbam = nn.ModuleList([CBAM(256) for _ in range(4)])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        f4 = self.fpn[3](features['stage4'])  # [32, 256, 7, 7]
        f4 = self.cbam[3](f4)
        f3 = self.fpn[2](features['stage3']) + f4  # [32, 256, 7, 7]
        f3 = self.cbam[2](f3)
        f2 = self.fpn[1](features['stage2']) + self.up(f3)  # [32, 256, 14, 14]
        f2 = self.cbam[1](f2)
        f1 = self.fpn[0](features['stage1']) + self.up(f2)  # [32, 256, 28, 28]
        f1 = self.cbam[0](f1)
        feats = [self.avg_pool(f).view(x.size(0), -1) for f in [f1, f2, f3, f4]]
        combined = torch.cat(feats, dim=1)
        return self.classifier(combined)

    def param_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.numel() for p in range(self.parameters()) if p.requires_grad)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params
        }


class GradCAM:
    def __init__(self, model, target_layers):
        """
        Initialize GradCAM with support for multiple target layers.
        Args:
            model: The model to analyze (e.g., SingleStreamEfficientCBAM)
            target_layers: A list of target layer names, e.g., ['features.2', 'features.4', 'features.6', 'features.8']
        """
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.model.eval()
        self.feature_extractor = create_feature_extractor(
            model.backbone,
            return_nodes={layer: layer for layer in target_layers}
        )

        def save_gradient(layer):
            def hook(grad):
                self.gradients[layer] = grad
            return hook

        def forward_hook(layer):
            def hook(module, input, output):
                self.activations[layer] = output
                if not output.requires_grad:
                    output.requires_grad_(True)
                output.register_hook(save_gradient(layer))
            return hook

        for param in model.backbone.parameters():
            param.requires_grad_(True)

        named_modules = dict(model.backbone.named_modules())
        for layer in target_layers:
            if layer not in named_modules:
                raise ValueError(f"Layer {layer} not found in model.backbone. Available layers: {list(named_modules.keys())}")
            target_module = named_modules[layer]
            print(f"Registering hook for layer {layer}")
            target_module.register_forward_hook(forward_hook(layer))

    def generate(self, input_image, target_classes=None):
        """
        Generate Grad-CAM heatmaps for specified classes, supporting multiple layers and multiple classes.
        Args:
            input_image: Input image tensor with shape [batch_size, channels, height, width].
            target_classes: List of target class indices. If None, select classes with predicted probability > 0.5.
        Returns:
            cams: A nested dictionary in the format {sample_idx: {layer: {class_idx: cam}}}.
            probs: Predicted probabilities for each class.
        """
        self.model.zero_grad()
        input_image = input_image.requires_grad_(True)
        features = self.feature_extractor(input_image)
        output = self.model(input_image)
        probs = torch.sigmoid(output)

        batch_size = input_image.size(0)
        if target_classes is None:
            target_classes = [torch.where(probs[i] > 0.5)[0].tolist() or [torch.argmax(probs[i]).item()]
                              for i in range(batch_size)]

        cams = {i: {layer: {} for layer in self.target_layers} for i in range(batch_size)}
        for i in range(batch_size):
            for target_class in target_classes[i]:
                self.model.zero_grad()
                one_hot = torch.zeros_like(output)
                one_hot[i, target_class] = 1
                output.backward(gradient=one_hot, retain_graph=True)
                for layer in self.target_layers:
                    if layer not in self.gradients or layer not in self.activations:
                        print(f"Debug: No gradients or activations for layer {layer}")
                        raise RuntimeError(f"No gradients or activations for layer {layer}")
                    gradients = self.gradients[layer][i:i+1]
                    activations = self.activations[layer][i:i+1]
                    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
                    cam = torch.sum(weights * activations, dim=1)
                    cam = F.relu(cam)
                    cam = cam - cam.min()
                    cam = cam / (cam.max() + 1e-8)
                    cams[i][layer][target_class] = cam.detach().cpu().numpy()
        return cams, probs.cpu().detach().numpy()

    def overlay_heatmap(self, cam, image, alpha=0.5):
        """
        Overlay the heatmap on the original image.
        Args:
            cam: The heatmap, with shape [height, width].
            image: The input image, with shape [batch_size, channels, height, width].
            alpha: The transparency of the overlay.
        Returns:
            overlay: The overlaid image, with shape [height, width, 3].
        """
        cam = cv2.resize(cam, (image.shape[3], image.shape[2]))
        cam = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        image = image.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)
        if image.shape[2] == 2:
            image = image[:, :, 0]
            image = np.stack([image] * 3, axis=-1)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = np.uint8(255 * image)
        overlay = (1 - alpha) * image + alpha * heatmap
        return np.uint8(overlay)

def plot_multi_layer_class_heatmaps(cams, image, img_name, layers, classes, defect_types, save_path, grad_cam):
    """
    Plot heatmaps for multiple layers and multiple classes, generating a single composite figure.
    Args:
        cams: Dictionary of heatmaps in the format {layer: {class_idx: cam}}.
        image: Input image with shape [batch_size, channels, height, width].
        img_name: Name of the image.
        layers: List of target layer names.
        classes: List of target class indices.
        defect_types: List of class names (defect type labels).
        save_path: Path to save the output figure.
        grad_cam: GradCAM object used for overlaying heatmaps.
    """
    n_rows = len(classes)
    n_cols = len(layers) + 1
    plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    for i, class_idx in enumerate(classes):
        plt.subplot(n_rows, n_cols, i * n_cols + 1)
        img = image.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)
        if img.shape[2] == 2:
            img = img[:, :, 0]
            img = np.stack([img] * 3, axis=-1)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = np.uint8(255 * img)
        plt.imshow(img)
        plt.title(f'Original Image\nClass: {defect_types[class_idx]}')
        plt.axis('off')

        for j, layer in enumerate(layers):
            plt.subplot(n_rows, n_cols, i * n_cols + j + 2)
            overlay = grad_cam.overlay_heatmap(cams[layer][class_idx][0], image)
            plt.imshow(overlay)
            plt.title(f'Layer: {layer}\nClass: {defect_types[class_idx]}')
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def evaluate_model(image_paths, labels, annotations, model_path, seed=None, is_ensemble=False, defect_types=None, config=None):
    """
        Evaluate the model and generate multi-layer, multi-class Grad-CAM heatmaps.
        Args:
            image_paths: List of image file paths in the test set.
            labels: Labels for the test set.
            annotations: Annotation data.
            model_path: Path to model weights (either a single path or a list of paths for ensemble models).
            seed: Random seed.
            is_ensemble: Whether the model is an ensemble.
            defect_types: List of class names (defect type labels).
            config: Configuration object.
        Returns:
            Dictionary containing evaluation results.
        """
    defect_types, defect_names_cn = get_defect_mapping(config)
    in_channels = 2 if config.use_binary_channel else 1
    test_dataset = AdvancedImageDataset(image_paths, labels, annotations, augment=False,
                                       use_binary_channel=config.use_binary_channel, config=config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    heatmap_batch_size = 4
    heatmap_loader = DataLoader(test_dataset, batch_size=heatmap_batch_size, shuffle=False, num_workers=2)

    target_layers = ['features.2.1.block.3', 'features.4.1.block.3', 'features.6.3.block.3', 'features.8']

    if is_ensemble:
        predictions = []  # list of (N_samples, num_classes) probs
        for path in model_path:
            model = SingleStreamEfficientCBAM(
                num_classes=config.num_classes,
                hidden_size=config.hidden_size,
                dropout_rate=config.dropout_rate
            ).to(config.device)
            sample_input = torch.randn(config.batch_size, in_channels, config.target_size[0], config.target_size[1]).to(
                config.device)
            try:
                output = model(sample_input)
                print(f"Output shape: {output.shape}")  # 预期为 [32, 6]
            except Exception as e:
                print(f"Forward pass test failed: {e}")
            model.load_state_dict(torch.load(path, map_location=config.device))
            model.eval()

            model_probs = []

            with torch.no_grad():
                for batch_idx, (images, _) in enumerate(test_loader):
                    images = images.to(config.device)
                    with autocast():
                        outputs = torch.sigmoid(model(images))
                    model_probs.append(outputs.cpu().numpy())

            predictions.append(np.concatenate(model_probs))
            del model
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Model deleted, memory cleared.")
        predictions = np.mean(predictions, axis=0)

        predicted_labels = np.zeros_like(predictions, dtype=int)
        for i in range(len(predictions)):
            if np.max(predictions[i]) < 0.5:
                max_idx = np.argmax(predictions[i])
                predicted_labels[i, max_idx] = 1
            else:
                predicted_labels[i] = (predictions[i] > 0.5).astype(int)
    else:
        heatmap_dir = os.path.join(config.save_root_dir, 'heatmaps_alone', f'seed_{seed}')
        os.makedirs(heatmap_dir, exist_ok=True)

        model = SingleStreamEfficientCBAM(
            num_classes=config.num_classes,
            hidden_size=config.hidden_size,
            dropout_rate=config.dropout_rate
        ).to(config.device)

        sample_input = torch.randn(config.batch_size, in_channels, config.target_size[0], config.target_size[1]).to(config.device)
        try:
            output = model(sample_input)
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Forward pass test failed: {e}")
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.eval()

        grad_cam = GradCAM(model, target_layers)
        predictions = []
        predicted_labels = []

        # 1. predict
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(test_loader):
                images = images.to(config.device)
                with autocast():
                    outputs = model(images)
                probs = torch.sigmoid(outputs)
                batch_probs = probs.cpu().numpy()
                predictions.append(batch_probs)

                batch_labels = np.zeros_like(batch_probs, dtype=int)
                for i in range(len(batch_probs)):
                    if np.max(batch_probs[i]) < 0.5:
                        max_idx = np.argmax(batch_probs[i])
                        batch_labels[i, max_idx] = 1
                    else:
                        batch_labels[i] = (batch_probs[i] > 0.5).astype(int)
                predicted_labels.append(batch_labels)

        # 2. heatmap
        for param in model.parameters():
            param.requires_grad = True
        for batch_idx, (images, _) in enumerate(heatmap_loader):
            images = images.to(config.device)
            images.requires_grad_(True)
            cams, batch_probs = grad_cam.generate(images)
            for i in range(images.size(0)):
                img_name = os.path.basename(image_paths[batch_idx * heatmap_batch_size + i])

                class_indices = list(cams[i][target_layers[0]].keys())
                if class_indices:
                    composite_save_path = os.path.join(heatmap_dir, f'composite_heatmap_{img_name}')
                    plot_multi_layer_class_heatmaps(
                        cams[i], images[i:i + 1], img_name, target_layers, class_indices, defect_types,
                        composite_save_path, grad_cam
                    )
            torch.cuda.empty_cache()
            gc.collect()
        print(f'Heatmaps saved successfully.')

        predictions = np.concatenate(predictions)
        predicted_labels = np.concatenate(predicted_labels)
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Model for seed {seed} deleted, memory cleared.")

    f1_micro = f1_score(labels, predicted_labels, average='micro')
    f1_per_class = f1_score(labels, predicted_labels, average=None)
    precision_micro = precision_score(labels, predicted_labels, average='micro')
    recall_micro = recall_score(labels, predicted_labels, average='micro')
    precision_per_class = precision_score(labels, predicted_labels, average=None, zero_division=0)
    recall_per_class = recall_score(labels, predicted_labels, average=None, zero_division=0)

    error_samples = [
        {
            "image_path": image_paths[i],
            "true_labels": labels[i].tolist(),
            "predicted_labels": predicted_labels[i].tolist(),
            "probabilities": predictions[i].tolist()
        } for i in range(len(labels)) if not np.array_equal(predicted_labels[i], labels[i])
    ]

    confusion_matrices = {defect_types[i]: confusion_matrix(labels[:, i], predicted_labels[:, i], labels=[0, 1]).tolist()
                         for i in range(config.num_classes)}

    pred_data = [
        {
            "image_path": image_paths[i],
            "true_labels": labels[i].tolist(),
            "predicted_labels": predicted_labels[i].tolist(),
            "probabilities": predictions[i].tolist()
        } for i in range(len(labels))
    ]
    os.makedirs(os.path.join(config.save_root_dir, 'test_results'), exist_ok=True)
    pred_path = os.path.join(config.save_root_dir, 'test_results',
                             f'predictions_ensemble.csv' if is_ensemble else f'predictions_seed{seed}.csv')
    pred_df = pd.DataFrame([
        {
            "Image_path": pred["image_path"],
            "True_labels": ",".join([defect_types[i] for i in range(config.num_classes) if pred["true_labels"][i]]),
            "Predicted_labels": ",".join(
                [defect_types[i] for i in range(config.num_classes) if pred["predicted_labels"][i]]),
            **{f"Prob_{defect_types[i]}": pred["probabilities"][i] for i in range(config.num_classes)}
        } for pred in pred_data
    ])
    pred_df.to_csv(pred_path, index=False, encoding='utf-8')
    if config.debug_mode:
        print(f"Saved predictions to {pred_path}")

    return {
        "f1_score": f1_micro,
        "f1_per_class": {defect_types[i]: f1 for i, f1 in enumerate(f1_per_class)},
        "precision": precision_micro,
        "recall": recall_micro,
        "precision_per_class": {defect_types[i]: float(p) for i, p in enumerate(precision_per_class)},
        "recall_per_class": {defect_types[i]: float(r) for i, r in enumerate(recall_per_class)},
        "error_samples": error_samples,
        "confusion_matrices": confusion_matrices
    }


def train_validate_test(input_config):
    global config
    config = input_config

    print(f"Starting multi-seed training with {len(config.random_seeds)} seeds in total.")

    ROOT_DIR = config.save_root_dir
    os.makedirs(ROOT_DIR, exist_ok=True)
    print(f"[Experiment directory locked] All results will be saved to: {ROOT_DIR}")

    os.makedirs(os.path.join(ROOT_DIR, "augmentation_samples"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "confusion_matrices"), exist_ok=True)

    defect_types, defect_names_cn = get_defect_mapping(config)
    start_time = datetime.datetime.now()
    process = psutil.Process()
    max_memory = 0

    log_data = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config.save_config(),
        "training": {},
        "test_results": {},
        "augmentation_stats": {},
        "label_inconsistencies": []
    }

    try:
        image_paths = [os.path.join(config.image_dir, f'{i}.png') for i in range(1, config.sample_num + 1)]
        labels = np.load(config.label_file)
        try:
            with open(config.annotation_file, 'r') as f:
                annotations = json.load(f)
        except Exception as e:
            print(f"Error: Failed to load annotation file {config.annotation_file}: {e}")
            return

        if config.debug_mode:
            print(f"Loaded {len(image_paths)} images from {config.image_dir}")
            print(f"Labels shape: {labels.shape}")
            print(f"Class distribution: {[np.sum(labels[:, i]) for i in range(config.num_classes)]}")

        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_seed)
        indices = np.arange(len(image_paths))
        for train_val_idx, test_idx in msss.split(image_paths, labels):
            test_indices = test_idx
            train_val_indices = train_val_idx
        test_paths = [image_paths[i] for i in test_indices]
        test_labels = labels[test_indices]
        train_val_paths = [image_paths[i] for i in train_val_indices]
        train_val_labels = labels[train_val_indices]
        if np.any(np.isin(test_indices, train_val_indices)):
            raise ValueError("Test set and train+validation set contain duplicate samples!")
        print('\n\n\n\n\n')
        print('==================================================')
        print(f'Initial data split:')
        print(f'test_indices: {test_indices}')
        print(f'train_val_indices: {train_val_indices}')
        print('==================================================')
        print('\n\n\n\n\n')

        print_dataset_distribution("test set", test_paths, test_labels)
        save_data_split([], [], [], [], test_paths, test_labels)  # 先保存测试集划分

        if config.debug_mode:
            print(f"Train + validation samples: {len(train_val_paths)}, test samples: {len(test_paths)}")
            print(f"Test set label distribution: {[np.sum(test_labels[:, i]) for i in range(config.num_classes)]}")

        all_best_val_f1_per_class = []

        total_seeds = len(config.random_seeds)

        for seed_idx, seed in enumerate(config.random_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            adjusted_val_size = config.val_size / (1 - config.test_size)
            msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=seed)
            train_val_indices = np.arange(len(train_val_paths))
            for train_idx, val_idx in msss.split(train_val_indices, train_val_labels):
                train_indices = train_idx
                val_indices = val_idx
            train_paths = [train_val_paths[i] for i in train_indices]
            train_labels = train_val_labels[train_indices]
            val_paths = [train_val_paths[i] for i in val_indices]
            val_labels = train_val_labels[val_indices]
            if np.any(np.isin(train_indices, val_indices)):
                raise ValueError("Training set and validation set contain duplicate samples!")
            print('\n\n\n\n\n')
            print('==================================================')
            print(f'Second data split:')
            print(f'val_indices: {val_indices}')
            print(f'train_indices: {train_indices}')
            print('==================================================')
            print('\n\n\n\n\n')

            print_dataset_distribution("train set", train_paths, train_labels)
            print_dataset_distribution("validation set", val_paths, val_labels)
            save_data_split(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, seed=seed)

            if config.debug_mode:
                print(f"Training samples: {len(train_paths)}, validation samples: {len(val_paths)}")
                print(f"Training set label distribution: {[np.sum(train_labels[:, i]) for i in range(config.num_classes)]}")
                print(f"Validation set label distribution: {[np.sum(val_labels[:, i]) for i in range(config.num_classes)]}")

            in_channels = 2 if config.use_binary_channel else 1
            model = SingleStreamEfficientCBAM(
                num_classes=config.num_classes,
                hidden_size=config.hidden_size,
                dropout_rate=config.dropout_rate
            ).to(config.device)
            sample_input = torch.randn(config.batch_size, 2, config.target_size[0], config.target_size[1]).to(
                config.device)
            output = model(sample_input)
            print(f"Model output shape: {output.shape}")
            train_dataset = AdvancedImageDataset(
                train_paths, train_labels, annotations, augment=True, model=model, use_binary_channel=config.use_binary_channel, config=config
            )
            val_dataset = AdvancedImageDataset(
                val_paths, val_labels, annotations, augment=False, use_binary_channel=config.use_binary_channel, config=config
            )
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                      pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8,
                                    pin_memory=True)
            log_data["augmentation_stats"][f"seed_{seed}"] = dict(train_dataset.augmentation_stats)
            criterion = MultiLabelFocalLoss(gamma=2.0, alpha=0.5)
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

            warmup_epochs = 5
            warmup_lr = config.learning_rate / 10
            best_val_f1 = 0
            early_stop_counter = 0
            train_loss_history = []
            val_loss_history = []
            train_f1_history = []
            val_f1_history = []
            train_precision_history = [[] for _ in range(config.num_classes)]
            train_recall_history = [[] for _ in range(config.num_classes)]
            train_f1_per_class_history = [[] for _ in range(config.num_classes)]
            val_precision_history = [[] for _ in range(config.num_classes)]
            val_recall_history = [[] for _ in range(config.num_classes)]
            val_f1_per_class_history = [[] for _ in range(config.num_classes)]
            training_log = {"epochs": [],
                            "best_val_f1": 0,
                            "best_epoch": 0,
                            "model_path": ""}

            best_val_f1_per_class = np.zeros(config.num_classes)
            lr_history = []
            best_epoch = 0
            val_error_samples = []

            for epoch in range(config.num_epochs):
                if epoch < warmup_epochs:
                    lr = warmup_lr + (config.learning_rate - warmup_lr) * (epoch / warmup_epochs)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = optimizer.param_groups[0]['lr']
                lr_history.append(lr)

                model.train()
                train_loss = 0
                train_preds = []
                train_true = []
                scaler = GradScaler()

                for images, labels in tqdm(train_loader, desc=f'HPLmodel {seed_idx+1}/{total_seeds} Seed {seed} epoch {epoch + 1}/{config.num_epochs}'):
                    images, labels = images.to(config.device), labels.to(config.device)
                    optimizer.zero_grad()
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss += loss.item()
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    train_preds.append(predicted.cpu().numpy())
                    train_true.append(labels.cpu().numpy())
                    mem_info = process.memory_info().rss / 1024 / 1024
                    max_memory = max(max_memory, mem_info)
                train_loss /= len(train_loader)
                train_preds = np.concatenate(train_preds)
                train_true = np.concatenate(train_true)
                train_f1 = f1_score(train_true, train_preds, average='micro')
                train_f1_per_class = f1_score(train_true, train_preds, average=None)
                train_precision_per_class = precision_score(train_true, train_preds, average=None, zero_division=0)
                train_recall_per_class = recall_score(train_true, train_preds, average=None, zero_division=0)

                train_loss_history.append(train_loss)
                train_f1_history.append(train_f1)

                for i in range(config.num_classes):
                    train_precision_history[i].append(train_precision_per_class[i])
                    train_recall_history[i].append(train_recall_per_class[i])
                    train_f1_per_class_history[i].append(train_f1_per_class[i])

                model.eval()
                val_loss = 0
                val_preds = []
                val_true = []
                val_predictions = []
                current_sample_idx = 0
                with torch.no_grad():
                    for batch_idx, (images, labels) in enumerate(val_loader):
                        images, labels = images.to(config.device), labels.to(config.device)
                        with autocast():
                            outputs = model(images)
                            val_loss += criterion(outputs, labels).item()
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                        val_preds.append(predicted.cpu().numpy())
                        val_true.append(labels.cpu().numpy())
                        probs = torch.sigmoid(outputs).cpu().numpy()
                        for i in range(len(labels)):
                            sample_idx = current_sample_idx + i
                            if sample_idx < len(val_paths):
                                val_predictions.append({
                                    "image_path": val_paths[sample_idx],
                                    "true_labels": labels[i].cpu().numpy().tolist(),
                                    "predicted_labels": predicted[i].cpu().numpy().tolist(),
                                    "probabilities": probs[i].tolist()
                                })
                                if not torch.equal(predicted[i], labels[i]):
                                    sample_idx = current_sample_idx + i
                                    if sample_idx < len(val_paths):
                                        val_error_samples.append({
                                            "index": sample_idx,
                                            "image_path": val_paths[sample_idx],
                                            "true_labels": labels[i].cpu().numpy().tolist(),
                                            "predicted_labels": predicted[i].cpu().numpy().tolist(),
                                            "probabilities": torch.sigmoid(outputs[i]).cpu().numpy().tolist()
                                        })
                            else:
                                print(f"Warning: Sample index {sample_idx} exceeds the length of val_paths ({len(val_paths)})")
                        current_sample_idx += len(labels)
                val_loss /= len(val_loader)
                val_preds = np.concatenate(val_preds)
                val_true = np.concatenate(val_true)
                val_f1 = f1_score(val_true, val_preds, average='micro')
                val_f1_per_class = f1_score(val_true, val_preds, average=None)
                val_loss_history.append(val_loss)
                val_f1_history.append(val_f1)
                val_precision_per_class = precision_score(val_true, val_preds, average=None, zero_division=0)
                val_recall_per_class = recall_score(val_true, val_preds, average=None, zero_division=0)
                for i in range(config.num_classes):
                    val_precision_history[i].append(val_precision_per_class[i])
                    val_recall_history[i].append(val_recall_per_class[i])
                    val_f1_per_class_history[i].append(val_f1_per_class[i])

                scheduler.step(val_loss)

                print(f'HPLmodel {seed_idx+1}/{total_seeds} Seed {seed}, Epoch {epoch + 1}/{config.num_epochs}:')
                print(f'  Training Loss: {train_loss:.4f} | F1 (micro): {train_f1:.4f}')
                print(f'  Validation Loss: {val_loss:.4f} | F1 (micro): {val_f1:.4f}')
                print(f'  Training Metrics (per class):')
                for i in range(config.num_classes):
                    print(f'{defect_names_cn[i]}: Precision={train_precision_per_class[i]:.4f}, Recall={train_recall_per_class[i]:.4f}, F1={train_f1_per_class[i]:.4f}')
                print(f'  Validation Metrics (per class):')
                for i in range(config.num_classes):
                    print(f'{defect_names_cn[i]}: Precision={val_precision_per_class[i]:.4f}, Recall={val_recall_per_class[i]:.4f}, F1={val_f1_per_class[i]:.4f}')
                print(f'  Current Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')

                epoch_log = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_f1": train_f1,
                    "train_f1_per_class": {defect_types[i]: f1 for i, f1 in enumerate(train_f1_per_class)},
                    "train_precision_per_class": {defect_types[i]: float(p) for i, p in
                                                  enumerate(train_precision_per_class)},
                    "train_recall_per_class": {defect_types[i]: float(r) for i, r in enumerate(train_recall_per_class)},
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "val_f1_per_class": {defect_types[i]: f1 for i, f1 in enumerate(val_f1_per_class)},
                    "val_precision_per_class": {defect_types[i]: float(p) for i, p in
                                                enumerate(val_precision_per_class)},
                    "val_recall_per_class": {defect_types[i]: float(r) for i, r in enumerate(val_recall_per_class)},
                    "learning_rate": lr,
                    "val_errors": val_error_samples.copy()
                }

                val_error_samples = []

                training_log["epochs"].append(epoch_log)

                if (epoch + 1) % 10 == 0:
                    checkpoint_path = os.path.join(ROOT_DIR,
                                                   f'checkpoint_seed{seed}_epoch{epoch + 1}.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    if config.debug_mode:
                        print(f"Save checkpoint to {checkpoint_path}")

                if val_f1 > best_val_f1:
                    best_epoch = epoch+1
                    best_val_f1 = val_f1
                    early_stop_counter = 0
                    model_path = os.path.join(ROOT_DIR, f'{config.save_model_prefix}_seed{seed}.pth')
                    torch.save(model.state_dict(), model_path)
                    training_log["model_path"] = model_path
                    training_log["best_epoch"] = epoch + 1
                    if config.debug_mode:
                        print(f"Save best model to {model_path}, best validation F1: {best_val_f1:.4f} at epoch {epoch + 1}")

                    os.makedirs(os.path.join(ROOT_DIR, 'metrics'), exist_ok=True)
                    pred_path = os.path.join(ROOT_DIR, 'metrics',
                                             f'val_predictions_seed{seed}_epoch{epoch + 1}.csv')
                    pred_data = []
                    for pred in val_predictions:
                        row = {
                            "Image_path": pred["image_path"],
                            "True_labels": ",".join(
                                [defect_types[i] for i in range(config.num_classes) if pred["true_labels"][i]]),
                            "Predicted_labels": ",".join([defect_types[i] for i in range(config.num_classes) if
                                                          pred["predicted_labels"][i]]),
                        }
                        for i in range(config.num_classes):
                            row[f"Prob_{defect_types[i]}"] = pred["probabilities"][i]
                        pred_data.append(row)
                    pd.DataFrame(pred_data).to_csv(pred_path, index=False, encoding='utf-8')
                    if config.debug_mode:
                        print(f"Saved validation predictions to {pred_path}")
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= config.early_stop_patience:
                        best_val_f1_per_class = val_f1_per_class
                        all_best_val_f1_per_class.append(best_val_f1_per_class.tolist())  # 转为list
                        print(f'Early stopping triggered at epoch {epoch + 1}; best epoch: {best_epoch}, '
                              f'validation macro F1: {best_val_f1:.4f}, '
                              f'validation F1 per class: {best_val_f1_per_class}')
                        break

                model_path = os.path.join(ROOT_DIR, f'{config.save_model_prefix}_seed{seed}.pth')
                torch.save(model.state_dict(), model_path)
                print(f"Force save the final model: {model_path}")

                if train_dataset.label_inconsistencies:
                    log_data["label_inconsistencies"].extend(train_dataset.label_inconsistencies)
                    train_dataset.label_inconsistencies = []
                    torch.cuda.empty_cache()
                    gc.collect()

            training_log["best_val_f1"] = best_val_f1
            log_data["training"][f"seed_{seed}"] = training_log

            os.makedirs(os.path.join(ROOT_DIR, 'metric_curves'), exist_ok=True)
            for i in range(config.num_classes):
                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.plot(range(1, len(train_precision_history[i]) + 1), train_precision_history[i],
                         label='train Precision')
                plt.plot(range(1, len(val_precision_history[i]) + 1), val_precision_history[i], label='val Precision')
                plt.xlabel('epoch')
                plt.ylabel('Precision')
                plt.title(f'{defect_names_cn[i]} Precision curve')
                plt.legend()
                plt.grid(True)

                plt.subplot(1, 3, 2)
                plt.plot(range(1, len(train_recall_history[i]) + 1), train_recall_history[i], label='train Recall')
                plt.plot(range(1, len(val_recall_history[i]) + 1), val_recall_history[i], label='val Recall')
                plt.xlabel('epoch')
                plt.ylabel('Recall')
                plt.title(f'{defect_names_cn[i]} Recall curve')
                plt.legend()
                plt.grid(True)

                plt.subplot(1, 3, 3)
                plt.plot(range(1, len(train_f1_per_class_history[i]) + 1), train_f1_per_class_history[i], label='train F1')
                plt.plot(range(1, len(val_f1_per_class_history[i]) + 1), val_f1_per_class_history[i], label='val F1')
                plt.xlabel('epoch')
                plt.ylabel('F1 value')
                plt.title(f'{defect_names_cn[i]} F1 curve')
                plt.legend()
                plt.grid(True)

                plt.tight_layout()
                save_path = os.path.join(ROOT_DIR, 'metric_curves',
                                         f'metric_curves_seed{seed}_{defect_types[i]}.png')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                if config.debug_mode:
                    print(f"Saving {defect_names_cn[i]} metric curves to {save_path}")

            os.makedirs(os.path.join(ROOT_DIR, 'training_curves'), exist_ok=True)
            plt.figure(figsize=(18, 5))
            plt.subplot(1, 3, 1)
            plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='train loss')
            plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='val loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title('loss curve')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(range(1, len(train_f1_history) + 1), train_f1_history, label='train F1 (micro)')
            plt.plot(range(1, len(val_f1_history) + 1), val_f1_history, label='val F1 (micro)')
            plt.xlabel('epoch')
            plt.ylabel('F1 value')
            plt.title('F1 value curve')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.plot(range(1, len(lr_history) + 1), lr_history, label='lr')
            plt.xlabel('epoch')
            plt.ylabel('lr')
            plt.title('lr curve')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            save_path = os.path.join(ROOT_DIR, 'training_curves', f'multilabel_train_curves_seed{seed}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            if config.debug_mode:
                print(f"Saving training curves to {save_path}")

            metrics_data = []
            for epoch_data in training_log["epochs"]:
                row = {
                    "Epoch": epoch_data["epoch"],
                    "Train_Loss": epoch_data["train_loss"],
                    "Train_F1": epoch_data["train_f1"],
                    "Val_Loss": epoch_data["val_loss"],
                    "Val_F1": epoch_data["val_f1"],
                    "Learning_Rate": epoch_data["learning_rate"]
                }
                for i in range(config.num_classes):
                    defect = defect_types[i]
                    row[f"Train_precision_{defect}"] = epoch_data["train_precision_per_class"][defect]
                    row[f"Train_recall_{defect}"] = epoch_data["train_recall_per_class"][defect]
                    row[f"Train_F1_{defect}"] = epoch_data["train_f1_per_class"][defect]
                    row[f"Val_precision_{defect}"] = epoch_data["val_precision_per_class"][defect]
                    row[f"Val_recall_{defect}"] = epoch_data["val_recall_per_class"][defect]
                    row[f"Val_F1_{defect}"] = epoch_data["val_f1_per_class"][defect]
                metrics_data.append(row)

            os.makedirs(os.path.join(ROOT_DIR, 'metrics'), exist_ok=True)
            metrics_path = os.path.join(ROOT_DIR, 'metrics', f'metrics_seed{seed}.csv')
            pd.DataFrame(metrics_data).to_csv(metrics_path, index=False, encoding='utf-8')
            if config.debug_mode:
                print(f"Saved metrics to {metrics_path}")

        # =================================================================================================================

        all_metrics = {seed: pd.read_csv(os.path.join(ROOT_DIR, 'metrics', f'metrics_seed{seed}.csv'))
                       for seed in config.random_seeds}
        min_epochs = min(len(df) for df in all_metrics.values())
        avg_metrics = []
        for epoch in range(min_epochs):
            row = {"Epoch": epoch + 1}
            for col in all_metrics[config.random_seeds[0]].columns:
                if col != "Epoch":
                    values = [df[col].iloc[epoch] for df in all_metrics.values()]
                    row[col] = np.mean(values)
            avg_metrics.append(row)

        avg_metrics_path = os.path.join(ROOT_DIR, 'metrics', 'avg_metrics.csv')
        pd.DataFrame(avg_metrics).to_csv(avg_metrics_path, index=False, encoding='utf-8')
        if config.debug_mode:
            print(f"Saved average metrics to {avg_metrics_path}")

        for i in range(config.num_classes):
            plt.figure(figsize=(15, 5))

            # Precision
            plt.subplot(1, 3, 1)
            plt.plot(range(1, min_epochs + 1),
                     [row[f'Train_precision_{defect_types[i]}'] for row in avg_metrics], label = 'Avg Train Precision')
            plt.plot(range(1, min_epochs + 1),
                     [row[f'Val_precision_{defect_types[i]}'] for row in avg_metrics], label = 'Avg Val Precision')

            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.title(f'{defect_names_cn[i]} Average Precision Curve')
            plt.legend()
            plt.grid(True)

            # Recall
            plt.subplot(1, 3, 2)
            plt.plot(range(1, min_epochs + 1),
                     [row[f'Train_recall_{defect_types[i]}'] for row in avg_metrics], label = 'Avg Train Recall')

            plt.plot(range(1, min_epochs + 1), [row[f'Val_recall_{defect_types[i]}'] for row in
                                                 avg_metrics], label = 'Avg Val Recall')

            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.title(f'{defect_names_cn[i]} Average Recall Curve')
            plt.legend()
            plt.grid(True)

            # F1
            plt.subplot(1, 3, 3)
            plt.plot(range(1, min_epochs + 1),
                     [row[f'Train_F1_{defect_types[i]}'] for row in avg_metrics], label = 'Avg Train F1')
            plt.plot(range(1, min_epochs + 1),
                     [row[f'Val_F1_{defect_types[i]}'] for row in avg_metrics], label = 'Avg Val F1')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.title(f'{defect_names_cn[i]} Average F1 Score')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            save_path = os.path.join(ROOT_DIR, 'metric_curves', f'avg_metric_curves_{defect_types[i]}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            if config.debug_mode:
                print(f"Saved average metric curve for {defect_types[i]} to {save_path})")

        error_stats = []
        for seed in config.random_seeds:
            seed_log = log_data["training"][f"seed_{seed}"]
            for epoch_data in seed_log["epochs"]:
                for err in epoch_data.get("val_errors", []):
                    true_labels = [defect_types[i] for i in range(config.num_classes) if err["true_labels"][i]]
                    pred_labels = [defect_types[i] for i in range(config.num_classes) if err["predicted_labels"][i]]
                    error_stats.append({
                        "Type": "Validation",
                        "Seed": seed,
                        "Epoch": epoch_data["epoch"],
                        "Image_path": err["image_path"],
                        "True_labels": ",".join(true_labels),
                        "Predicted_labels": ",".join(pred_labels),
                        "Max_prob": max(err["probabilities"]),
                        "Min_prob": min(err["probabilities"])
                    })
        os.makedirs(os.path.join(ROOT_DIR, 'val_error_analysis'), exist_ok=True)
        error_stats_path = os.path.join(ROOT_DIR, 'val_error_analysis', 'val_error_stats.csv')
        pd.DataFrame(error_stats).to_csv(error_stats_path, index=False, encoding='utf-8')
        if config.debug_mode:
            print(f"Saved error statistics to {error_stats_path}")

        best_val_f1_file = os.path.join(ROOT_DIR, 'metrics', 'best_val_f1_per_class.json')
        os.makedirs(os.path.dirname(best_val_f1_file), exist_ok=True)
        best_val_f1_data = {
            "seeds": config.random_seeds,
            "best_val_f1_per_class": all_best_val_f1_per_class,
            "defect_types": [defect_types[i] for i in range(config.num_classes)]
        }
        with open(best_val_f1_file, 'w', encoding='utf-8') as f:
            json.dump(best_val_f1_data, f, indent=4, ensure_ascii=False)
        if config.debug_mode:
            print(f"Saving best validation F1 score to {best_val_f1_file}")

        # ====================================================test==================================================================
        single_test_results = {}
        for seed in config.random_seeds:
            model_path = os.path.join(ROOT_DIR, f'{config.save_model_prefix}_seed{seed}.pth')
            single_test_results[f"seed_{seed}"] = evaluate_model(test_paths, test_labels, annotations, model_path, defect_types=defect_types, seed=seed, config=config)
            os.makedirs(os.path.join(ROOT_DIR, 'confusion_matrices'), exist_ok=True)
            for i in range(config.num_classes):
                cm = np.array(single_test_results[f"seed_{seed}"]["confusion_matrices"][defect_types[i]])
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['-', '+'], yticklabels=['-', '+'])
                plt.title(f'{defect_types[i]} confusion matrices (Seed {seed})')
                plt.xlabel('predicted')
                plt.ylabel('true')
                save_path = os.path.join(ROOT_DIR, 'confusion_matrices',
                                         f'cm_seed{seed}_{defect_types[i]}.png')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                if config.debug_mode:
                    print(f"Saving {defect_names_cn[i]} confusion matrix to {save_path}")
            print(f'Seed {seed} model test results:')
            print(f'  F1 Score (micro): {single_test_results[f"seed_{seed}"]["f1_score"]:.4f}')
            print(f'  Precision (micro): {single_test_results[f"seed_{seed}"]["precision"]:.4f}')
            print(f'  Recall (micro): {single_test_results[f"seed_{seed}"]["recall"]:.4f}')
            print(f'  Per-class metrics:')
            for i in range(config.num_classes):
                print(f'{defect_names_cn[i]}: Precision={single_test_results[f"seed_{seed}"]["precision_per_class"][defect_types[i]]:.4f}, Recall={single_test_results[f"seed_{seed}"]["recall_per_class"][defect_types[i]]:.4f}, F1={single_test_results[f"seed_{seed}"]["f1_per_class"][defect_types[i]]:.4f}')
            log_data[f"seed_{seed}_model_test_results"] = {
                "test_f1_score": single_test_results[f"seed_{seed}"]["f1_score"],
                "test_f1_per_class": single_test_results[f"seed_{seed}"]["f1_per_class"],
                "test_precision": single_test_results[f"seed_{seed}"]["precision"],
                "test_recall": single_test_results[f"seed_{seed}"]["recall"],
                "test_precision_per_class": single_test_results[f"seed_{seed}"]["precision_per_class"],
                "test_recall_per_class": single_test_results[f"seed_{seed}"]["recall_per_class"],
                "test_error_samples": single_test_results[f"seed_{seed}"]["error_samples"]
            }
            single_error_dir = os.path.join(ROOT_DIR, f"seed_{seed}_model_error_samples")
            os.makedirs(single_error_dir, exist_ok=True)
            for err in single_test_results[f"seed_{seed}"]["error_samples"]:
                img = io.imread(err["image_path"], as_gray=True)
                true_labels = [defect_types[i] for i in range(config.num_classes) if err["true_labels"][i] == 1]
                pred_labels = [defect_types[i] for i in range(config.num_classes) if err["predicted_labels"][i] == 1]
                plt.imshow(img, cmap='gray')
                plt.title(f'true label: {", ".join(true_labels)}\npredicted label: {", ".join(pred_labels)}')
                plt.axis('off')
                save_path = os.path.join(single_error_dir, f'error_{os.path.basename(err["image_path"])}')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                err["image_save_path"] = save_path

        model_paths = [os.path.join(ROOT_DIR, f'{config.save_model_prefix}_seed{seed}.pth') for seed in config.random_seeds]
        test_results = evaluate_model(test_paths, test_labels, annotations, model_paths, is_ensemble=True, defect_types=defect_types, config=config)
        os.makedirs(os.path.join(ROOT_DIR, 'confusion_matrices'), exist_ok=True)
        for i in range(config.num_classes):
            cm = np.array(test_results["confusion_matrices"][defect_types[i]])
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['-', '+'], yticklabels=['-', '+'])
            plt.title(f'{defect_types[i]} confusion_matrices (ensemble prediction)')
            plt.xlabel('predicted')
            plt.ylabel('true')
            save_path = os.path.join(ROOT_DIR, 'confusion_matrices', f'cm_ensemble_{defect_types[i]}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            if config.debug_mode:
                print(f"Saving {defect_names_cn[i]} confusion matrix to {save_path}")
        print(f'Ensemble test results:')
        print(f'  F1 Score (micro): {test_results["f1_score"]:.4f}')
        print(f'  Precision (micro): {test_results["precision"]:.4f}')
        print(f'  Recall (micro): {test_results["recall"]:.4f}')
        print(f'  Per-class metrics:')
        for i in range(config.num_classes):
            print(
                f'    {defect_names_cn[i]}: Precision={test_results["precision_per_class"][defect_types[i]]:.4f}, Recall={test_results["recall_per_class"][defect_types[i]]:.4f}, F1={test_results["f1_per_class"][defect_types[i]]:.4f}')
        log_data["ensemble_test_results"] = {
            "test_f1_score": test_results["f1_score"],
            "test_f1_per_class": test_results["f1_per_class"],
            "test_precision": test_results["precision"],
            "test_recall": test_results["recall"],
            "test_precision_per_class": test_results["precision_per_class"],
            "test_recall_per_class": test_results["recall_per_class"],
            "test_error_samples": test_results["error_samples"]
        }
        ensemble_error_dir = os.path.join(ROOT_DIR, "ensemble_error_samples")
        os.makedirs(ensemble_error_dir, exist_ok=True)
        for err in test_results["error_samples"]:
            img = io.imread(err["image_path"], as_gray=True)
            true_labels = [defect_types[i] for i in range(config.num_classes) if err["true_labels"][i] == 1]
            pred_labels = [defect_types[i] for i in range(config.num_classes) if err["predicted_labels"][i] == 1]
            plt.imshow(img, cmap='gray')
            plt.title(f'true label: {", ".join(true_labels)}\npredicted label: {", ".join(pred_labels)}')
            plt.axis('off')
            save_path = os.path.join(ensemble_error_dir, f'error_{os.path.basename(err["image_path"])}')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            err["image_save_path"] = save_path

        test_metrics = []
        for seed in config.random_seeds:
            res = single_test_results[f"seed_{seed}"]
            row = {
                "Model": f"Seed_{seed}",
                "F1_score": res["f1_score"],
                "Precision": res["precision"],
                "Recall": res["recall"]
            }
            for i in range(config.num_classes):
                defect = defect_types[i]
                row[f"F1_{defect}"] = res["f1_per_class"][defect]
                row[f"Precision_{defect}"] = res["precision_per_class"][defect]
                row[f"Recall_{defect}"] = res["recall_per_class"][defect]
            test_metrics.append(row)

        row = {
            "Model": "Ensemble",
            "F1_score": test_results["f1_score"],
            "Precision": test_results["precision"],
            "Recall": test_results["recall"]
        }
        for i in range(config.num_classes):
            defect = defect_types[i]
            row[f"F1_{defect}"] = test_results["f1_per_class"][defect]
            row[f"Precision_{defect}"] = test_results["precision_per_class"][defect]
            row[f"Recall_{defect}"] = test_results["recall_per_class"][defect]
        test_metrics.append(row)

        os.makedirs(os.path.join(ROOT_DIR, 'test_results'), exist_ok=True)
        metrics_path = os.path.join(ROOT_DIR, 'test_results', 'test_metrics.csv')
        pd.DataFrame(test_metrics).to_csv(metrics_path, index=False, encoding='utf-8')
        if config.debug_mode:
            print(f"Saved test metrics to {metrics_path}")

        cm_data = []
        for seed in config.random_seeds:
            res = single_test_results[f"seed_{seed}"]
            for i in range(config.num_classes):
                defect = defect_types[i]
                cm = np.array(res["confusion_matrices"][defect])

                if cm.shape == (1, 1):
                    tn = cm[0, 0]
                    fp = 0
                    fn = 0
                    tp = 0
                else:
                    tn = cm[0, 0]
                    fp = cm[0, 1]
                    fn = cm[1, 0]
                    tp = cm[1, 1]

                cm_data.append({
                    "Model": f"Seed_{seed}",
                    "Defect": defect,
                    "TN": tn,
                    "FP": fp,
                    "FN": fn,
                    "TP": tp
                })

        for i in range(config.num_classes):
            defect = defect_types[i]
            cm = np.array(test_results["confusion_matrices"][defect])

            if cm.shape == (1, 1):
                tn = cm[0, 0]
                fp = 0
                fn = 0
                tp = 0
            else:
                tn = cm[0, 0]
                fp = cm[0, 1]
                fn = cm[1, 0]
                tp = cm[1, 1]

            cm_data.append({
                "Model": "Ensemble",
                "Defect": defect,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp
            })
        os.makedirs(os.path.join(ROOT_DIR, 'test_results'), exist_ok=True)
        cm_path = os.path.join(ROOT_DIR, 'test_results', 'confusion_matrices.csv')
        pd.DataFrame(cm_data).to_csv(cm_path, index=False, encoding='utf-8')
        if config.debug_mode:
            print(f"Saved confusion matrices to {cm_path}")

        error_stats = []
        for seed in config.random_seeds:
            for err in log_data[f"seed_{seed}_model_test_results"]["test_error_samples"]:
                true_labels = [defect_types[i] for i in range(config.num_classes) if err["true_labels"][i]]
                pred_labels = [defect_types[i] for i in range(config.num_classes) if err["predicted_labels"][i]]
                error_stats.append({
                    "Type": "Test",
                    "Seed": seed,
                    "Epoch": 0,
                    "Image_path": err["image_path"],
                    "True_labels": ",".join(true_labels),
                    "Predicted_labels": ",".join(pred_labels),
                    "Max_prob": max(err["probabilities"]),
                    "Min_prob": min(err["probabilities"])
                })
        for err in log_data["ensemble_test_results"]["test_error_samples"]:
            true_labels = [defect_types[i] for i in range(config.num_classes) if err["true_labels"][i]]
            pred_labels = [defect_types[i] for i in range(config.num_classes) if err["predicted_labels"][i]]
            error_stats.append({
                "Type": "Test_Ensemble",
                "Seed": 0,
                "Epoch": 0,
                "Image_path": err["image_path"],
                "True_labels": ",".join(true_labels),
                "Predicted_labels": ",".join(pred_labels),
                "Max_prob": max(err["probabilities"]),
                "Min_prob": min(err["probabilities"])
            })

        os.makedirs(os.path.join(ROOT_DIR, 'test_error_analysis'), exist_ok=True)
        error_stats_path = os.path.join(ROOT_DIR, 'test_error_analysis', 'test_error_stats.csv')
        pd.DataFrame(error_stats).to_csv(error_stats_path, index=False, encoding='utf-8')
        if config.debug_mode:
            print(f"Saved error statistics to {error_stats_path}")

        plt.figure(figsize=(10, 6))
        f1_scores = [log_data[f"seed_{seed}_model_test_results"]["test_f1_score"] for seed in config.random_seeds]
        f1_scores.append(log_data["ensemble_test_results"]["test_f1_score"])
        labels = [f"Seed {seed}" for seed in config.random_seeds] + ["Ensemble"]
        plt.bar(labels, f1_scores, color=['#1f77b4'] * len(config.random_seeds) + ['#ff7f0e'])
        plt.ylabel('F1 Score (micro)')
        plt.title('Test F1 Scores by Model')
        plt.grid(True, axis='y')
        os.makedirs(os.path.join(ROOT_DIR, 'summary'), exist_ok=True)
        save_path = os.path.join(ROOT_DIR, 'summary', 'test_f1_scores.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        if config.debug_mode:
            print(f"Saved test F1 scores plot to {save_path}")

        # ======================================================================================================================

        def convert_to_json_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            return obj

        end_time = datetime.datetime.now()
        log_data_new = {
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": (datetime.datetime.now() - start_time).total_seconds(),
            "max_memory_mb": max_memory,
            "config": config.save_config(),
            "summary": {},
            "seeds": {},
            "data_split_file": config.data_split_file,
            "metrics_files": [os.path.join(ROOT_DIR, 'metrics', f'metrics_seed{seed}.csv')
                              for seed in config.random_seeds] + [avg_metrics_path],
            "test_metrics_file": metrics_path,
            "confusion_matrices_file": cm_path,
            "error_stats_file": error_stats_path
        }
        for seed in config.random_seeds:
            seed_log = log_data["training"][f"seed_{seed}"]
            log_data_new["seeds"][str(seed)] = {
                "best_val_f1": seed_log["best_val_f1"],
                "best_epoch": seed_log["best_epoch"],
                "model_path": seed_log["model_path"],
                "test_results": log_data[f"seed_{seed}_model_test_results"]
            }
        test_f1_scores = [log_data[f"seed_{seed}_model_test_results"]["test_f1_score"]
                          for seed in config.random_seeds]
        log_data_new["summary"] = {
            "avg_test_f1": float(np.mean(test_f1_scores)),
            "best_seed": config.random_seeds[np.argmax(test_f1_scores)],
            "best_test_f1": float(np.max(test_f1_scores)),
            "avg_test_f1_per_class": {
                defect_types[i]: float(
                    np.mean([log_data[f"seed_{seed}_model_test_results"]["test_f1_per_class"][defect_types[i]]
                             for seed in config.random_seeds]))
                for i in range(config.num_classes)
            }
        }
        log_data_new = convert_to_json_serializable(log_data_new)
        os.makedirs(ROOT_DIR, exist_ok=True)
        with open(config.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data_new, f, indent=4, ensure_ascii=False)
        if config.debug_mode:
            print(f"Saving training log to {config.log_file}")

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        torch.cuda.empty_cache()
        gc.collect()
        raise

    finally:
        del train_loader, val_loader, train_dataset, val_dataset
        gc.collect()
        torch.cuda.empty_cache()


def test_only(input_config,model_path=None):
    global config
    config = input_config

    ROOT_DIR = config.save_root_dir
    os.makedirs(ROOT_DIR, exist_ok=True)
    print(f"[Experiment directory locked] All results will be saved to: {ROOT_DIR}")

    defect_types, defect_names_cn = get_defect_mapping(config)

    start_time = datetime.datetime.now()
    log_data = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "single_model_results": {},
        "ensemble_results": {},
        "test_results": {}
    }

    image_paths = [os.path.join(config.image_dir, f'{i}.png') for i in range(1, config.sample_num + 1)]
    labels = np.load(config.label_file)
    try:
        with open(config.annotation_file, 'r') as f:
            annotations = json.load(f)
    except Exception as e:
        print(f"Error: Failed to load annotation file {config.annotation_file}: {e}")
        return

    if config.debug_mode:
        print(f"Loaded {len(image_paths)} images from {config.image_dir}")
        print(f"Label shape: {labels.shape}")
        print(f"Class distribution: {[np.sum(labels[:, i]) for i in range(config.num_classes)]}")

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_seed)
    indices = np.arange(len(image_paths))
    for train_val_idx, test_idx in msss.split(image_paths, labels):
        test_indices = test_idx
        train_val_indices = train_val_idx
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = labels[test_indices]
    train_val_paths = [image_paths[i] for i in train_val_indices]
    train_val_labels = labels[train_val_indices]
    if np.any(np.isin(test_indices, train_val_indices)):
        raise ValueError("Test set and train+validation set contain duplicate samples!")
    print('\n\n\n\n\n')
    print('==================================================')
    print(f'Initial data split:')
    print(f'test_indices: {test_indices}')
    print(f'train_val_indices: {train_val_indices}')
    print('==================================================')
    print('\n\n\n\n\n')

    print_dataset_distribution("test set", test_paths, test_labels)
    save_data_split([], [], [], [], test_paths, test_labels)

    if config.debug_mode:
        print(f"Train + validation samples: {len(train_val_paths)}, test samples: {len(test_paths)}")

    model_paths = [os.path.join(ROOT_DIR, f'{config.save_model_prefix}_seed{seed}.pth')
                   for seed in config.random_seeds]
    if config.debug_mode:
        print(f"Loading model(s) for evaluation: {model_paths}")

    for model_path in model_paths:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist")

    single_test_results = {}
    for seed in config.random_seeds:
        model_path = os.path.join(ROOT_DIR, f'{config.save_model_prefix}_seed{seed}.pth')

        single_test_results[f"seed_{seed}"] = evaluate_model(test_paths, test_labels, annotations, model_path, defect_types=defect_types, seed=seed, config=config)
        os.makedirs(os.path.join(ROOT_DIR, 'confusion_matrices'), exist_ok=True)
        for i in range(config.num_classes):
            cm = np.array(single_test_results[f"seed_{seed}"]["confusion_matrices"][defect_types[i]])
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['-', '+'], yticklabels=['-', '+'])
            plt.title(f'{defect_types[i]} confusion matrices (Seed {seed})')
            plt.xlabel('predicted')
            plt.ylabel('true')
            save_path = os.path.join(ROOT_DIR, 'confusion_matrices',
                                     f'cm_seed{seed}_{defect_types[i]}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            if config.debug_mode:
                print(f"Saving {defect_names_cn[i]} confusion matrix to {save_path}")
        print(f'Seed {seed} model test results:')
        print(f'  F1 Score (micro): {single_test_results[f"seed_{seed}"]["f1_score"]:.4f}')
        print(f'  Precision (micro): {single_test_results[f"seed_{seed}"]["precision"]:.4f}')
        print(f'  Recall (micro): {single_test_results[f"seed_{seed}"]["recall"]:.4f}')
        print(f'  Per-class metrics:')
        for i in range(config.num_classes):
            print(
                f'{defect_names_cn[i]}: Precision={single_test_results[f"seed_{seed}"]["precision_per_class"][defect_types[i]]:.4f}, Recall={single_test_results[f"seed_{seed}"]["recall_per_class"][defect_types[i]]:.4f}, F1={single_test_results[f"seed_{seed}"]["f1_per_class"][defect_types[i]]:.4f}')
        log_data[f"seed_{seed}_model_test_results"] = {
            "test_f1_score": single_test_results[f"seed_{seed}"]["f1_score"],
            "test_f1_per_class": single_test_results[f"seed_{seed}"]["f1_per_class"],
            "test_precision": single_test_results[f"seed_{seed}"]["precision"],
            "test_recall": single_test_results[f"seed_{seed}"]["recall"],
            "test_precision_per_class": single_test_results[f"seed_{seed}"]["precision_per_class"],
            "test_recall_per_class": single_test_results[f"seed_{seed}"]["recall_per_class"],
            "test_error_samples": single_test_results[f"seed_{seed}"]["error_samples"]
        }
        single_error_dir = os.path.join(ROOT_DIR, f"seed_{seed}_model_error_samples")
        os.makedirs(single_error_dir, exist_ok=True)
        for err in single_test_results[f"seed_{seed}"]["error_samples"]:
            img = io.imread(err["image_path"], as_gray=True)
            true_labels = [defect_types[i] for i in range(config.num_classes) if err["true_labels"][i] == 1]
            pred_labels = [defect_types[i] for i in range(config.num_classes) if
                           err["predicted_labels"][i] == 1]
            plt.imshow(img, cmap='gray')
            plt.title(f'true label: {", ".join(true_labels)}\npredicted label: {", ".join(pred_labels)}')
            plt.axis('off')
            save_path = os.path.join(single_error_dir, f'error_{os.path.basename(err["image_path"])}')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            err["image_save_path"] = save_path

    model_paths = [os.path.join(ROOT_DIR, f'{config.save_model_prefix}_seed{seed}.pth') for seed in
                   config.random_seeds]
    test_results = evaluate_model(test_paths, test_labels, annotations, model_paths, is_ensemble=True, defect_types=defect_types, config=config)

    os.makedirs(os.path.join(ROOT_DIR, 'confusion_matrices'), exist_ok=True)
    for i in range(config.num_classes):
        cm = np.array(test_results["confusion_matrices"][defect_types[i]])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['-', '+'], yticklabels=['-', '+'])
        plt.title(f'{defect_types[i]} confusion_matrices (ensemble prediction)')
        plt.xlabel('predicted')
        plt.ylabel('true')
        save_path = os.path.join(ROOT_DIR, 'confusion_matrices',
                                 f'cm_ensemble_{defect_types[i]}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        if config.debug_mode:
            print(f"Saving {defect_names_cn[i]} confusion matrix to {save_path}")
    print(f'Ensemble test results:')
    print(f'  F1 Score (micro): {test_results["f1_score"]:.4f}')
    print(f'  Precision (micro): {test_results["precision"]:.4f}')
    print(f'  Recall (micro): {test_results["recall"]:.4f}')
    print(f'  Per-class metrics:')
    for i in range(config.num_classes):
        print(
            f'    {defect_names_cn[i]}: Precision={test_results["precision_per_class"][defect_types[i]]:.4f}, Recall={test_results["recall_per_class"][defect_types[i]]:.4f}, F1={test_results["f1_per_class"][defect_types[i]]:.4f}')
    log_data["ensemble_test_results"] = {
        "test_f1_score": test_results["f1_score"],
        "test_f1_per_class": test_results["f1_per_class"],
        "test_precision": test_results["precision"],
        "test_recall": test_results["recall"],
        "test_precision_per_class": test_results["precision_per_class"],
        "test_recall_per_class": test_results["recall_per_class"],
        "test_error_samples": test_results["error_samples"]
    }
    ensemble_error_dir = os.path.join(ROOT_DIR, "ensemble_error_samples")
    os.makedirs(ensemble_error_dir, exist_ok=True)
    for err in test_results["error_samples"]:
        img = io.imread(err["image_path"], as_gray=True)
        true_labels = [defect_types[i] for i in range(config.num_classes) if err["true_labels"][i] == 1]
        pred_labels = [defect_types[i] for i in range(config.num_classes) if err["predicted_labels"][i] == 1]
        plt.imshow(img, cmap='gray')
        plt.title(f'true label: {", ".join(true_labels)}\npredicted label: {", ".join(pred_labels)}')
        plt.axis('off')
        save_path = os.path.join(ensemble_error_dir, f'error_{os.path.basename(err["image_path"])}')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        err["image_save_path"] = save_path

    test_metrics = []
    for seed in config.random_seeds:
        res = single_test_results[f"seed_{seed}"]
        row = {
            "Model": f"Seed_{seed}",
            "F1_score": res["f1_score"],
            "Precision": res["precision"],
            "Recall": res["recall"]
        }
        for i in range(config.num_classes):
            defect = defect_types[i]
            row[f"F1_{defect}"] = res["f1_per_class"][defect]
            row[f"Precision_{defect}"] = res["precision_per_class"][defect]
            row[f"Recall_{defect}"] = res["recall_per_class"][defect]
        test_metrics.append(row)

    row = {
        "Model": "Ensemble",
        "F1_score": test_results["f1_score"],
        "Precision": test_results["precision"],
        "Recall": test_results["recall"]
    }
    for i in range(config.num_classes):
        defect = defect_types[i]
        row[f"F1_{defect}"] = test_results["f1_per_class"][defect]
        row[f"Precision_{defect}"] = test_results["precision_per_class"][defect]
        row[f"Recall_{defect}"] = test_results["recall_per_class"][defect]
    test_metrics.append(row)

    os.makedirs(os.path.join(ROOT_DIR, 'test_results'), exist_ok=True)
    metrics_path = os.path.join(ROOT_DIR, 'test_results', 'test_metrics.csv')
    pd.DataFrame(test_metrics).to_csv(metrics_path, index=False, encoding='utf-8')
    if config.debug_mode:
        print(f"Saved test metrics to {metrics_path}")

    cm_data = []
    for seed in config.random_seeds:
        res = single_test_results[f"seed_{seed}"]
        for i in range(config.num_classes):
            defect = defect_types[i]
            cm = np.array(res["confusion_matrices"][defect])
            cm_data.append({
                "Model": f"Seed_{seed}",
                "Defect": defect,
                "TN": cm[0, 0],
                "FP": cm[0, 1],
                "FN": cm[1, 0],
                "TP": cm[1, 1]
            })
    for i in range(config.num_classes):
        defect = defect_types[i]
        cm = np.array(test_results["confusion_matrices"][defect])
        cm_data.append({
            "Model": "Ensemble",
            "Defect": defect,
            "TN": cm[0, 0],
            "FP": cm[0, 1],
            "FN": cm[1, 0],
            "TP": cm[1, 1]
        })
    os.makedirs(os.path.join(ROOT_DIR, 'test_results'), exist_ok=True)
    cm_path = os.path.join(ROOT_DIR, 'test_results', 'confusion_matrices.csv')
    pd.DataFrame(cm_data).to_csv(cm_path, index=False, encoding='utf-8')
    if config.debug_mode:
        print(f"Saved confusion matrices to {cm_path}")

    error_stats = []
    for seed in config.random_seeds:
        for err in log_data[f"seed_{seed}_model_test_results"]["test_error_samples"]:
            true_labels = [defect_types[i] for i in range(config.num_classes) if err["true_labels"][i]]
            pred_labels = [defect_types[i] for i in range(config.num_classes) if err["predicted_labels"][i]]
            error_stats.append({
                "Type": "Test",
                "Seed": seed,
                "Epoch": 0,
                "Image_path": err["image_path"],
                "True_labels": ",".join(true_labels),
                "Predicted_labels": ",".join(pred_labels),
                "Max_prob": max(err["probabilities"]),
                "Min_prob": min(err["probabilities"])
            })
    for err in log_data["ensemble_test_results"]["test_error_samples"]:
        true_labels = [defect_types[i] for i in range(config.num_classes) if err["true_labels"][i]]
        pred_labels = [defect_types[i] for i in range(config.num_classes) if err["predicted_labels"][i]]
        error_stats.append({
            "Type": "Test_Ensemble",
            "Seed": 0,
            "Epoch": 0,
            "Image_path": err["image_path"],
            "True_labels": ",".join(true_labels),
            "Predicted_labels": ",".join(pred_labels),
            "Max_prob": max(err["probabilities"]),
            "Min_prob": min(err["probabilities"])
        })

    os.makedirs(os.path.join(ROOT_DIR, 'test_error_analysis'), exist_ok=True)
    error_stats_path = os.path.join(ROOT_DIR, 'test_error_analysis', 'test_error_stats.csv')
    pd.DataFrame(error_stats).to_csv(error_stats_path, index=False, encoding='utf-8')
    if config.debug_mode:
        print(f"Saved error statistics to {error_stats_path}")

    plt.figure(figsize=(10, 6))
    f1_scores = [log_data[f"seed_{seed}_model_test_results"]["test_f1_score"] for seed in config.random_seeds]
    f1_scores.append(log_data["ensemble_test_results"]["test_f1_score"])
    labels = [f"Seed {seed}" for seed in config.random_seeds] + ["Ensemble"]
    plt.bar(labels, f1_scores, color=['#1f77b4'] * len(config.random_seeds) + ['#ff7f0e'])
    plt.ylabel('F1 Score (micro)')
    plt.title('Test F1 Scores by Model')
    plt.grid(True, axis='y')
    os.makedirs(os.path.join(ROOT_DIR, 'summary'), exist_ok=True)
    save_path = os.path.join(ROOT_DIR, 'summary', 'test_f1_scores.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    if config.debug_mode:
        print(f"Saved test F1 scores plot to {save_path}")

    def convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        return obj

    end_time = datetime.datetime.now()
    log_data_new = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "config": config.save_config(),
        "summary": {
            "avg_test_f1": float(np.mean([log_data[f"seed_{seed}_model_test_results"]["test_f1_score"]
                                          for seed in config.random_seeds])),
            "best_seed": config.random_seeds[np.argmax([log_data[f"seed_{seed}_model_test_results"]["test_f1_score"]
                                                        for seed in config.random_seeds])],
            "best_test_f1": float(np.max([log_data[f"seed_{seed}_model_test_results"]["test_f1_score"]
                                          for seed in config.random_seeds])),
            "ensemble_test_f1": log_data["ensemble_test_results"]["test_f1_score"]
        },
        "seeds": {
            str(seed): {
                "model_path": os.path.join(ROOT_DIR, f'{config.save_model_prefix}_seed{seed}.pth'),
                "test_results": log_data[f"seed_{seed}_model_test_results"]
            } for seed in config.random_seeds
        },
        "ensemble_results": log_data["ensemble_test_results"],
        "test_metrics_file": metrics_path,
        "confusion_matrices_file": cm_path,
        "error_stats_file": error_stats_path
    }
    log_data_new = convert_to_json_serializable(log_data_new)
    test_log_file = os.path.join(ROOT_DIR, "test_log.json")
    os.makedirs(ROOT_DIR, exist_ok=True)
    with open(test_log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data_new, f, indent=4, ensure_ascii=False)
    if config.debug_mode:
        print(f"Saving test log to {test_log_file}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()

    config = Config(
        label_file="your_labels.npy",
        num_classes=4,
    )
    print("Warning: You are running the training script directly. Please launch via the GUI to access full functionality!")

    if args.mode == 'train':
        train_validate_test(config)
    else:
        test_only(config)
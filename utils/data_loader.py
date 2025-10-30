import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import pydicom  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pydicom = None


@dataclass
class DataConfig:
    train_dir: str
    test_dir: str
    val_split: float
    class_names: List[str]
    allowed_exts: List[str]
    color_mode: str
    target_size: Tuple[int, int]
    dicom_enabled: bool
    batch_size: int
    shuffle: bool


def _is_allowed(path: str, exts: List[str]) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in [e.lower() for e in exts]


def _read_image(path: str, target_size: Tuple[int, int], color_mode: str) -> np.ndarray:
    if path.lower().endswith('.dcm'):
        if pydicom is None:
            raise ImportError("pydicom is required to read DICOM files")
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        if cv2 is None:
            raise ImportError("opencv-python is required for resizing images")
        if color_mode == 'rgb':
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        arr = cv2.resize(arr, target_size, interpolation=cv2.INTER_AREA)
        return arr

    if cv2 is None:
        raise ImportError("opencv-python is required for reading images")
    flag = cv2.IMREAD_COLOR if color_mode == 'rgb' else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    if color_mode == 'rgb' and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def _discover_classes(root_dir: str) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    names: List[str] = []
    for entry in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, entry)
        if os.path.isdir(full):
            names.append(entry)
    return names


def load_config(yaml_path: str = 'config/data.yaml') -> DataConfig:
    import yaml  # local import to keep base import light
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get('paths', {})
    images = cfg.get('images', {})
    dicom = cfg.get('dicom', {})

    class_names: List[str] = cfg.get('class_names') or _discover_classes(paths.get('train_dir', 'data/train'))
    if not class_names:
        class_names = []

    return DataConfig(
        train_dir=paths.get('train_dir', 'data/train'),
        test_dir=paths.get('test_dir', 'data/test'),
        val_split=float(paths.get('val_split', 0.15)),
        class_names=class_names,
        allowed_exts=images.get('allowed_exts', ['.jpg', '.jpeg', '.png', '.bmp', '.dcm']),
        color_mode=images.get('color_mode', 'rgb'),
        target_size=tuple(images.get('target_size', [224, 224])),
        dicom_enabled=bool(dicom.get('enabled', True)),
        batch_size=int(cfg.get('loader', {}).get('batch_size', 32)),
        shuffle=bool(cfg.get('loader', {}).get('shuffle', True)),
    )


def index_dataset(split_dir: str, class_names: List[str], allowed_exts: List[str]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {c: [] for c in class_names}
    if not os.path.isdir(split_dir):
        return index
    for class_name in class_names:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for root, _, files in os.walk(class_dir):
            for fn in files:
                path = os.path.join(root, fn)
                if _is_allowed(path, allowed_exts):
                    index[class_name].append(path)
    return index


def train_val_split(train_index: Dict[str, List[str]], val_ratio: float, shuffle: bool = True) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    train_out: Dict[str, List[str]] = {}
    val_out: Dict[str, List[str]] = {}
    for c, items in train_index.items():
        items_copy = list(items)
        if shuffle:
            random.shuffle(items_copy)
        n_val = int(len(items_copy) * val_ratio)
        val_out[c] = items_copy[:n_val]
        train_out[c] = items_copy[n_val:]
    return train_out, val_out


def summarize_index(name: str, index: Dict[str, List[str]]) -> Dict[str, int]:
    counts = {k: len(v) for k, v in index.items()}
    counts['__total__'] = sum(counts.values())
    return counts


__all__ = [
    'DataConfig',
    'load_config',
    'index_dataset',
    'train_val_split',
    'summarize_index',
]



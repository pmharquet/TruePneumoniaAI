import os
import random
import cv2
import numpy as np


class Dataset:
    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        for image, label in self._items:
            yield image.astype(np.float64), label

    def shuffle(self):
        random.shuffle(self._items)


class DatasetLoader:
    CLASSES = {'normal': 0, 'bacteria': 1, 'virus': 2}

    def __init__(self, train_dir, val_dir, target_size=None, max_images=None):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.target_size = target_size
        self.max_images = max_images
        self._train = None
        self._val = None

    def _extract_label(self, filename):
        fname = filename.lower()
        for cls, idx in self.CLASSES.items():
            if cls in fname:
                return idx
        return None

    def _load_dir(self, directory, split_name):
        items = []
        files = sorted(f for f in os.listdir(directory) if f.lower().endswith('.jpg'))
        if self.max_images is not None:
            files = files[:self.max_images]

        unknown = 0
        total = len(files)
        print(f"  Chargement {split_name} : {total} images…")

        for i, filename in enumerate(files):
            label = self._extract_label(filename)
            if label is None:
                unknown += 1
                continue

            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            if self.target_size is not None:
                image = cv2.resize(image, (self.target_size[1], self.target_size[0]))

            items.append((image, label))

            if (i + 1) % 1000 == 0 or (i + 1) == total:
                print(f"    {i + 1}/{total}", end='\r')

        print()
        if unknown:
            print(f"  ({unknown} fichiers ignorés — label inconnu)")
        return items

    def load(self):
        self._train = Dataset(self._load_dir(self.train_dir, "Train"))
        self._val   = Dataset(self._load_dir(self.val_dir,   "Val"))
        print(f"  Train : {len(self._train)} images  |  Val : {len(self._val)} images")
        return self

    def get_train(self):
        return self._train

    def get_val(self):
        return self._val

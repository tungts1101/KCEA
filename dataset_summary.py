import numpy as np
import sys
import os
from io import StringIO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_TABLE
from data_manager import DataManager

SEED = 1993

for dataset_name, [(num_tasks, init_cls, increment)] in DATA_TABLE.items():
    _stdout = sys.stdout
    sys.stdout = StringIO()
    dm = DataManager(dataset_name, shuffle=True, seed=SEED, init_cls=init_cls, increment=increment)
    sys.stdout = _stdout

    targets = dm._train_targets
    classes, counts = np.unique(targets, return_counts=True)
    total_classes = len(classes)
    total_samples = len(targets)

    print(f"\n{'='*50}")
    print(f"Dataset : {dataset_name}")
    print(f"  Total training samples : {total_samples}")
    print(f"  Total classes          : {total_classes}")
    print(f"  Samples per class      :")
    for cls, cnt in zip(classes, counts):
        print(f"    Class {cls:3d}: {cnt}")

"""Synthetic datasets for Zero-Shot Learning demos."""

from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticZSLDataset(Dataset):
    """Synthetic dataset for Zero-Shot Learning.

    A class is defined by a set of binary attributes. The neural perception
    receives noisy observations of these attributes and the PSL reasoning
    layer propagates logical constraints to predict unobserved attributes.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_obs_attrs : int
        Number of observed attributes (used as input features to perception).
    n_unobs_attrs : int
        Number of unobserved attributes (predicted by the PSL reasoning layer).
    n_classes : int
        Number of classes in the dataset.
    classes_to_use : list of int, optional
        Which class indices to include. Used to simulate zero-shot by excluding some.
    seed : int
        Random seed for reproducibility.
    """

    OBSERVED_ATTR_NAMES = [
        "has_wings", "has_fur", "has_scales", "has_gills",
        "has_legs", "has_wheels", "has_feathers", "has_tail",
        "can_fly", "can_swim", "lives_in_water", "lives_in_air",
    ]
    UNOBSERVED_ATTR_NAMES = [
        "is_bird", "is_mammal", "is_fish", "is_reptile",
        "is_insect", "is_amphibian", "is_flying_animal", "is_water_animal",
    ]

    def __init__(
        self,
        n_samples: int,
        n_obs_attrs: int = 8,
        n_unobs_attrs: int = 4,
        n_classes: int = 6,
        classes_to_use: Optional[List[int]] = None,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.n_obs_attrs = n_obs_attrs
        self.n_unobs_attrs = n_unobs_attrs
        self.n_classes = n_classes

        np.random.seed(seed)

        # Attribute signatures per class
        self.class_attr_map = np.zeros(
            (n_classes, n_obs_attrs + n_unobs_attrs), dtype=np.float32
        )

        for c in range(n_classes):
            if c == 0:  # bird-like
                self.class_attr_map[c, 0] = 1  # has_wings
                self.class_attr_map[c, 6] = 1  # has_feathers
                self.class_attr_map[c, 8] = 1  # can_fly
                self.class_attr_map[c, n_obs_attrs + 0] = 1  # is_bird
            elif c == 1:  # mammal-like
                self.class_attr_map[c, 1] = 1  # has_fur
                self.class_attr_map[c, 4] = 1  # has_legs
                self.class_attr_map[c, n_obs_attrs + 1] = 1  # is_mammal
            elif c == 2:  # fish-like
                self.class_attr_map[c, 3] = 1  # has_gills
                self.class_attr_map[c, 5] = 1  # lives_in_water
                self.class_attr_map[c, n_obs_attrs + 2] = 1  # is_fish
            elif c == 3:  # reptile-like
                self.class_attr_map[c, 2] = 1  # has_scales
                self.class_attr_map[c, 4] = 1  # has_legs
                self.class_attr_map[c, n_obs_attrs + 3] = 1  # is_reptile
            elif c == 4:  # insect-like
                self.class_attr_map[c, 0] = 1  # has_wings
                self.class_attr_map[c, 8] = 1  # can_fly
                self.class_attr_map[c, n_obs_attrs + 4] = 1  # is_insect
            else:  # amphibian-like
                self.class_attr_map[c, 1] = 1
                self.class_attr_map[c, 5] = 1
                self.class_attr_map[c, n_obs_attrs + 7] = 1

        self.class_attr_map = np.clip(self.class_attr_map, 0, 1).astype(np.float32)

        if classes_to_use is None:
            self.classes_to_use = list(range(n_classes))
        else:
            self.classes_to_use = classes_to_use

        self.observed_attr_names = self.OBSERVED_ATTR_NAMES[:n_obs_attrs]
        self.unobserved_attr_names = self.UNOBSERVED_ATTR_NAMES[:n_unobs_attrs]

        # Pre-generate data
        self.data: np.ndarray = np.zeros((n_samples, n_obs_attrs), dtype=np.float32)
        self.labels: np.ndarray = np.zeros((n_samples, n_classes), dtype=np.float32)
        self.true_classes: np.ndarray = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            c = np.random.choice(self.classes_to_use)
            self.true_classes[i] = c
            noise = np.random.normal(0, 0.15, n_obs_attrs)
            self.data[i] = np.clip(
                self.class_attr_map[c, :n_obs_attrs] + noise, -0.3, 1.3
            ).astype(np.float32)
            target = np.zeros(n_classes, dtype=np.float32)
            target[c] = 1.0
            self.labels[i] = target

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.data[idx]),
            torch.tensor(self.labels[idx]),
        )

    def get_true_classes(self) -> np.ndarray:
        """Return true class labels for all samples."""
        return self.true_classes.copy()

    def get_rule_definitions(self) -> list:
        """Return PSL rule definitions for zero-shot learning demo.

        Rules connect observed attributes -> unobserved (logical) attributes.
        """
        return [
            {"body": ["has_wings", "has_feathers"], "head": ["is_bird"], "weight": 2.0},
            {"body": ["has_fur", "has_legs"], "head": ["is_mammal"], "weight": 2.0},
            {"body": ["has_scales"], "head": ["is_reptile"], "weight": 1.5},
            {"body": ["has_gills"], "head": ["is_fish"], "weight": 2.0},
            {"body": ["has_wings", "can_fly"], "head": ["is_insect"], "weight": 1.0},
            {"body": ["has_fur"], "head": ["is_water_animal"], "weight": 0.5},
        ]

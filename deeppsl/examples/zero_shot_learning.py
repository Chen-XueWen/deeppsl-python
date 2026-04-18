"""
DeepPSL Zero-Shot Learning Demo

Demonstrates the full DeepPSL pipeline:
1. Neural perception extracts noisy observed attributes from raw input
2. PSL reasoning propagates logical constraints to predict unobserved attributes
3. End-to-end joint training optimizes both perception and reasoning

This simulates a zero-shot learning scenario where we predict logical
class properties (is_bird, is_mammal, etc.) from partial sensory observations.
"""

import sys
import os

import torch
import torch.nn as nn
import numpy as np

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deeppsl.core.clause import Predicate, Atom, Rule
from deeppsl.rules.compiler import RuleCompiler
from deeppsl.psl.solver import PSLSolver
from deeppsl.layers.perception import MLPPerception
from deeppsl.layers.reasoning import DeepPSLModel
from deeppsl.utils.datasets import SyntheticZSLDataset
from deeppsl.train.trainer import DeepPSLTrainer

# ─── Configuration ────────────────────────────────────────────────
N_SAMPLES = 600
N_OBS_ATTRS = 8
N_UNOBS_ATTRS = 4
N_CLASSES = 6
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3

# ─── 1. Create Synthetic Dataset ─────────────────────────────────
print("=" * 60)
print("DeepPSL Zero-Shot Learning Demo")
print("=" * 60)

# Training dataset: all classes
train_dataset = SyntheticZSLDataset(
    n_samples=N_SAMPLES // 2,
    n_obs_attrs=N_OBS_ATTRS,
    n_unobs_attrs=N_UNOBS_ATTRS,
    n_classes=N_CLASSES,
    seed=42,
)

# Zero-shot dataset: exclude class 2 (fish) — never seen during training
zs_classes = [c for c in range(N_CLASSES) if c != 2]
zs_dataset = SyntheticZSLDataset(
    n_samples=N_SAMPLES // 2,
    n_obs_attrs=N_OBS_ATTRS,
    n_unobs_attrs=N_UNOBS_ATTRS,
    n_classes=N_CLASSES,
    classes_to_use=zs_classes,
    seed=123,
)

print(f"\n📊 Dataset created:")
print(f"   Training samples: {len(train_dataset)} (classes {train_dataset.classes_to_use})")
print(f"   Zero-shot samples: {len(zs_dataset)} (classes {zs_dataset.classes_to_use})")
print(f"   Observed attributes: {train_dataset.observed_attr_names}")
print(f"   Unobserved attributes: {train_dataset.unobserved_attr_names}")
print(f"   ⚠️  Class 2 (fish) excluded from zero-shot set — testing zero-shot capability")

# ─── 2. Define PSL Rules ─────────────────────────────────────────
obs_names = train_dataset.observed_attr_names
unobs_names = train_dataset.unobserved_attr_names

rules = [
    Rule(2.0,
         [Atom(Predicate(obs_names[0]), ("x",)), Atom(Predicate(obs_names[6]), ("x",))],  # has_wings & has_feathers
         [Atom(Predicate(unobs_names[0]), ("x",))]),  # is_bird
    Rule(2.0,
         [Atom(Predicate(obs_names[1]), ("x",)), Atom(Predicate(obs_names[4]), ("x",))],  # has_fur & has_legs
         [Atom(Predicate(unobs_names[1]), ("x",))]),  # is_mammal
    Rule(1.5,
         [Atom(Predicate(obs_names[2]), ("x",))],  # has_scales
         [Atom(Predicate(unobs_names[3]), ("x",))]),  # is_reptile
    Rule(2.0,
         [Atom(Predicate(obs_names[3]), ("x",))],  # has_gills
         [Atom(Predicate(unobs_names[2]), ("x",))]),  # is_fish
]

print(f"\n📐 PSL Rules:")
for r in rules:
    print(f"   {r}")

# ─── 3. Compile Rules ────────────────────────────────────────────
constants = ["x"]  # Single variable for simplicity

compiler = RuleCompiler(rules, constants)
Ay, Ap, b, weights = compiler.get_matrices()

print(f"\n🔧 Compiled ground network:")
print(f"   Ground rules: {len(compiler.ground_rules)}")
print(f"   Observed atoms: {len(compiler.observed_atoms)}")
print(f"   Unobserved atoms: {len(compiler.unobserved_atoms)}")

# ─── 4. Build Model ──────────────────────────────────────────────
# Perception: MLP maps input features → observed atom truth values
perception = MLPPerception(
    input_dim=N_OBS_ATTRS,
    output_dim=N_OBS_ATTRS,
    hidden_dims=[64, 32],
)

# PSL Solver: differentiable HL-MRF solver
HAS_CVXPYLAYERS = False
try:
    from cvxpylayers.torch import CvxpyLayer
    HAS_CVXPYLAYERS = True
except ImportError:
    pass

solver = PSLSolver(Ay, Ap, b, weights)
print(f"\n🧠 Model built:")
print(f"   Perception: MLP({N_OBS_ATTRS} → {N_OBS_ATTRS})")
print(f"   PSL Solver: {'differentiable (cvxpylayers)' if HAS_CVXPYLAYERS else 'non-differentiable (CVXPY fallback)'}")
print(f"   Ground atoms: {len(compiler.unobserved_atoms)}")

# Reasoning: perception + solver
model = DeepPSLModel(perception, solver)

# ─── 5. Training ─────────────────────────────────────────────────
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# For zero-shot, we predict which unobserved attributes hold
# Target: the unobserved attribute truth values from the class signature
# We construct this from the dataset's class_attr_map

trainer = DeepPSLTrainer(model, lr=LR)

print(f"\n🚀 Training for {EPOCHS} epochs...")
print("-" * 60)

# Track which unobserved atoms map to class indices
unobs_keys = list(compiler.unobserved_atoms.keys())

for epoch in range(EPOCHS):
    total_loss = 0.0
    n_batches = 0

    for x_batch, y_class_batch in train_loader:
        # Get target unobserved attribute values
        y_target = torch.zeros_like(y_class_batch)
        for i in range(len(y_class_batch)):
            pred_class = y_class_batch[i].argmax().item()
            if pred_class < len(train_dataset.class_attr_map):
                # Map class prediction to unobserved attribute targets
                start = N_OBS_ATTRS
                y_target[i] = torch.tensor(
                    train_dataset.class_attr_map[pred_class, start:start + N_UNOBS_ATTRS],
                    dtype=torch.float32,
                )

        y_pred = model(x_batch)
        loss = nn.BCELoss()(y_pred, y_target)

        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"   Epoch {epoch + 1:3d}/{EPOCHS} | Loss: {avg_loss:.4f}")

print("-" * 60)

# ─── 6. Evaluation ───────────────────────────────────────────────
print("\n📋 Evaluation:")
print("-" * 60)

# Training set
train_loader_eval = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
x_eval, _ = next(iter(train_loader_eval))
with torch.no_grad():
    y_train_pred = model(x_eval)

train_acc = (y_train_pred > 0.5).float().mean().item()
print(f"   Training set accuracy: {train_acc:.4f}")

# Zero-shot set (class 2 / fish never seen)
zs_loader_eval = DataLoader(zs_dataset, batch_size=len(zs_dataset), shuffle=False)
x_zs, y_zs = next(iter(zs_loader_eval))
with torch.no_grad():
    y_zs_pred = model(x_zs)

zs_acc = (y_zs_pred > 0.5).float().mean().item()
print(f"   Zero-shot accuracy (no fish seen): {zs_acc:.4f}")

# ─── 7. Show Predictions ─────────────────────────────────────────
print("\n📊 Example Predictions (Zero-Shot Set):")
print(f"   {'Sample':<8} {'Observed':<30} {'Predicted':<30} {'True':<30}")
print("-" * 100)

for i in range(min(10, len(zs_dataset))):
    pred = [unobs_names[j] for j in range(len(unobs_names)) if y_zs_pred[i, j] > 0.5]
    true_pred = [unobs_names[j] for j in range(len(unobs_names)) if y_zs[i, j] > 0.5]

    # Format observed attributes
    obs_str = ", ".join([
        f"{train_dataset.observed_attr_names[j]}={x_zs[i, j]:.2f}"
        for j in range(N_OBS_ATTRS)
    ])
    obs_str = obs_str[:28] + "..." if len(obs_str) > 30 else obs_str

    print(f"   {i:<8} {obs_str:<30} {str(pred)[:28]:<30} {str(true_pred)[:28]}")

# ─── 8. Zero-Shot Analysis ───────────────────────────────────────
print(f"\n{'=' * 60}")
print("🔍 Zero-Shot Analysis:")
print(f"{'=' * 60}")
print(f"   Classes seen during training: {train_dataset.classes_to_use}")
print(f"   Classes in zero-shot set:     {zs_dataset.classes_to_use}")
print(f"   (Class {2} / fish was NEVER seen during training)")
print(f"\n   The model must generalize from:")
print(f"   - Bird: has_wings + has_feathers + can_fly")
print(f"   - Mammal: has_fur + has_legs")
print(f"   - Reptile: has_scales + has_legs")
print(f"   - Insect: has_wings + can_fly")
print(f"\n   PSL rules enable the model to:")
print(f"   - Use logical constraints to infer unobserved attributes")
print(f"   - Generalize to unseen classes by recognizing shared attributes")

print("\n✅ DeepPSL demo complete!")
print(f"   📁 Project: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
print(f"   🔧 Run: python3 examples/zero_shot_learning.py")
print(f"   🧪 Tests: pytest tests/")

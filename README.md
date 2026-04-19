# DeepPSL: Differentiable Probabilistic Soft Logic

DeepPSL is a neuro-symbolic AI framework that integrates deep learning perception with Probabilistic Soft Logic (PSL) reasoning. It treats PSL inference as a convex optimization problem (HL-MRF) and makes it differentiable using `cvxpylayers`.

Demo Page: https://deeppsl-demo.vercel.app/, Original Paper: https://arxiv.org/abs/2109.13662

## Architecture

- **Perception Layer**: Neural networks (CNN/MLP) that map raw input to continuous truth values in [0, 1] for observed predicates.
- **Reasoning Layer**: A PSL solver that takes observed truth values and infers truth values for unobserved predicates by minimizing rule violations.
- **Differentiable Solver**: Uses CVXPY and `cvxpylayers` to backpropagate gradients from reasoning outcomes back to perception parameters.

## Installation

```bash
pip install torch numpy cvxpy cvxpylayers
```

Note: `cvxpylayers` requires Python development headers to build `diffcp`.

## Usage

### 1. Define Rules and Compiling

```python
from deeppsl.core.clause import Predicate, Atom, Rule
from deeppsl.rules.compiler import RuleCompiler

# Define predicates
has_wings = Predicate("has_wings", is_observed=True)
is_bird = Predicate("is_bird", is_observed=False)

# Define rule: is_bird(X) -> has_wings(X)
rule = Rule(10.0, [Atom(is_bird, ("X",))], [Atom(has_wings, ("X",))])

# Compile for a domain
compiler = RuleCompiler([rule], ["entity_1"])
Ay, Ap, b, weights = compiler.get_matrices()
```

### 2. End-to-End Model

```python
from deeppsl.psl.solver import PSLSolver
from deeppsl.layers.reasoning import DeepPSLModel

solver = PSLSolver(Ay, Ap, b, weights)
model = DeepPSLModel(perception_nn, solver)

# Forward pass
# x: input features
# y: inferred truth values for class predicates
y = model(x)
```

## Examples

Run the Zero-Shot Learning demo:
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 examples/zero_shot_learning.py
```

## Testing

```bash
pytest tests/
```

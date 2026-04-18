import torch
from torch.utils.data import DataLoader
from deeppsl.core.clause import Predicate, Atom, Rule
from deeppsl.rules.compiler import RuleCompiler
from deeppsl.psl.solver import PSLSolver, HAS_CVXPYLAYERS
from deeppsl.layers.perception import MLPPerception
from deeppsl.layers.reasoning import DeepPSLModel
from deeppsl.train.trainer import DeepPSLTrainer
from deeppsl.utils.datasets import SyntheticZSLDataset

def main():
    # 1. Configuration
    n_attrs = 8
    n_classes = 4
    batch_size = 16
    epochs = 10
    
    # 2. Setup Data
    train_dataset = SyntheticZSLDataset(200, n_attrs, n_classes, classes_to_use=[0, 1, 2])
    test_dataset = SyntheticZSLDataset(100, n_attrs, n_classes, classes_to_use=[0, 1, 2, 3])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 3. Define Logic Rules
    attr_preds = [Predicate(f"attr_{i}", is_observed=True) for i in range(n_attrs)]
    class_preds = [Predicate(f"class_{j}", is_observed=False) for j in range(n_classes)]
    
    rules = []
    class_attr_map = train_dataset.class_attr_map
    for c in range(n_classes):
        for k in range(n_attrs):
            if class_attr_map[c, k] == 1:
                rules.append(Rule(10.0, [Atom(class_preds[c], ("X",))], [Atom(attr_preds[k], ("X",))]))
            else:
                rules.append(Rule(10.0, [Atom(class_preds[c], ("X",)), Atom(attr_preds[k], ("X",))], []))
    
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            rules.append(Rule(5.0, [Atom(class_preds[i], ("X",)), Atom(class_preds[j], ("X",))], []))

    # 4. Compile Rules
    print("Compiling rules...")
    compiler = RuleCompiler(rules, ["X"])
    Ay, Ap, b, weights = compiler.get_matrices()
    print(f"Ground Network: {Ay.shape[0]} clauses, {Ay.shape[1]} unobserved atoms.")

    # 5. Initialize Model
    perception = MLPPerception(n_attrs, n_attrs)
    solver = PSLSolver(Ay, Ap, b, weights)
    model = DeepPSLModel(perception, solver)
    
    # 6. Training / Evaluation
    if HAS_CVXPYLAYERS:
        trainer = DeepPSLTrainer(model, lr=0.01)
        print("\nStarting Training (cvxpylayers found)...")
        for epoch in range(1, epochs + 1):
            total_loss = 0
            for x, y in train_loader:
                total_loss += trainer.train_step(x, y)
            if epoch % 5 == 0 or epoch == 1:
                val_loss, val_acc = trainer.evaluate(test_loader)
                print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {val_acc:.4f}")
    else:
        print("\nSkipping training (cvxpylayers missing). Performing Zero-Shot evaluation directly.")
        print("Note: In ZSL, the rules guide the inference even without training on target classes.")
        # Without training, we assume perception is identity (features are noisy attributes)
        # We can simulate this by setting weights to identity
        with torch.no_grad():
            for layer in perception.model:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.eye_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
        
        trainer = DeepPSLTrainer(model)
        val_loss, val_acc = trainer.evaluate(test_loader)
        print(f"Zero-Shot Evaluation | Test Acc: {val_acc:.4f}")

    # 7. Final Evaluation on Unseen Class
    print("\nEvaluating specifically on Unseen Class (Class 3)...")
    unseen_dataset = SyntheticZSLDataset(50, n_attrs, n_classes, classes_to_use=[3])
    unseen_loader = DataLoader(unseen_dataset, batch_size=batch_size)
    _, unseen_acc = trainer.evaluate(unseen_loader)
    print(f"Unseen Class Accuracy: {unseen_acc:.4f}")

if __name__ == "__main__":
    main()

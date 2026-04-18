import numpy as np
import itertools
from typing import List, Dict, Tuple
from deeppsl.core.clause import Rule, Atom, Predicate

class RuleCompiler:
    """Compiles First-Order Logic PSL rules into matrix form for optimization."""
    def __init__(self, rules: List[Rule], constants: List[str]):
        self.rules = rules
        self.constants = constants
        self.ground_rules: List[Rule] = []
        self.observed_atoms: Dict[Tuple, int] = {}  # (pred_name, args) -> idx
        self.unobserved_atoms: Dict[Tuple, int] = {}  # (pred_name, args) -> idx
        self._compile()

    def _compile(self):
        # 1. Ground all rules by substituting variables with constants
        for rule in self.rules:
            vars_in_rule = sorted(list({arg for atom in rule.body + rule.head 
                                      for arg in atom.arguments if arg[0].isupper()}))
            
            if not vars_in_rule:
                self.ground_rules.append(rule)
                continue
            
            for vals in itertools.product(self.constants, repeat=len(vars_in_rule)):
                var_map = dict(zip(vars_in_rule, vals))
                g_body = [self._ground_atom(a, var_map) for a in rule.body]
                g_head = [self._ground_atom(a, var_map) for a in rule.head]
                self.ground_rules.append(Rule(rule.weight, g_body, g_head))

        # 2. Collect all unique ground atoms and assign indices
        for g_rule in self.ground_rules:
            for atom in g_rule.body + g_rule.head:
                key = (atom.predicate.name, atom.arguments)
                if atom.predicate.is_observed:
                    if key not in self.observed_atoms:
                        self.observed_atoms[key] = len(self.observed_atoms)
                else:
                    if key not in self.unobserved_atoms:
                        self.unobserved_atoms[key] = len(self.unobserved_atoms)

    def _ground_atom(self, atom: Atom, var_map: Dict[str, str]) -> Atom:
        g_args = tuple(var_map.get(arg, arg) for arg in atom.arguments)
        return Atom(atom.predicate, g_args)

    def get_matrices(self):
        """Returns Ay, Ap, b, and weights for the ground network."""
        nc = len(self.ground_rules)
        ny = len(self.unobserved_atoms)
        np_ = len(self.observed_atoms)
        
        Ay = np.zeros((nc, ny))
        Ap = np.zeros((nc, np_))
        b = np.zeros(nc)
        weights = np.zeros(nc)

        for i, rule in enumerate(self.ground_rules):
            weights[i] = rule.weight
            # Violation: max(0, sum(body) - sum(head) + 1 - |body|)
            b[i] = 1.0 - len(rule.body)
            for atom in rule.body:
                key = (atom.predicate.name, atom.arguments)
                if atom.predicate.is_observed:
                    Ap[i, self.observed_atoms[key]] = 1.0
                else:
                    Ay[i, self.unobserved_atoms[key]] = 1.0
            for atom in rule.head:
                key = (atom.predicate.name, atom.arguments)
                if atom.predicate.is_observed:
                    Ap[i, self.observed_atoms[key]] = -1.0
                else:
                    Ay[i, self.unobserved_atoms[key]] = -1.0
        
        return Ay, Ap, b, weights

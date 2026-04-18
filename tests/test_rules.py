import pytest
import numpy as np
from deeppsl.core.clause import Predicate, Atom, Rule
from deeppsl.rules.compiler import RuleCompiler

def test_compiler_grounding():
    p1 = Predicate("p1", is_observed=True, arity=1)
    p2 = Predicate("p2", is_observed=False, arity=1)
    # rule: p1(X) -> p2(X)
    rule = Rule(1.0, [Atom(p1, ("X",))], [Atom(p2, ("X",))])
    
    compiler = RuleCompiler([rule], ["a", "b"])
    assert len(compiler.ground_rules) == 2
    assert len(compiler.observed_atoms) == 2 # p1(a), p1(b)
    assert len(compiler.unobserved_atoms) == 2 # p2(a), p2(b)

def test_compiler_matrices():
    p1 = Predicate("p1", is_observed=True)
    p2 = Predicate("p2", is_observed=False)
    # p1(X) & p2(X) -> False  (Violation: p1 + p2 - 0 - (2-1) = p1 + p2 - 1)
    rule = Rule(1.0, [Atom(p1, ("X",)), Atom(p2, ("X",))], [])
    compiler = RuleCompiler([rule], ["a"])
    Ay, Ap, b, weights = compiler.get_matrices()
    
    # max(0, 1.0*p1 + 1.0*p2 - 1.0)
    assert Ay[0, 0] == 1.0
    assert Ap[0, 0] == 1.0
    assert b[0] == -1.0

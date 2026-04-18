from typing import List, Tuple, Union

class Predicate:
    """Represents a logical predicate."""
    def __init__(self, name: str, is_observed: bool = False, arity: int = 1):
        self.name = name
        self.is_observed = is_observed
        self.arity = arity

    def __repr__(self):
        return f"Predicate({self.name}, obs={self.is_observed})"

class Atom:
    """Represents a grounded or ungrounded atom (predicate applied to arguments)."""
    def __init__(self, predicate: Predicate, arguments: Tuple[str, ...]):
        if len(arguments) != predicate.arity:
            raise ValueError(f"Predicate {predicate.name} expects {predicate.arity} arguments, got {len(arguments)}")
        self.predicate = predicate
        self.arguments = arguments

    def is_ground(self) -> bool:
        # For simplicity, assume arguments starting with uppercase are variables
        return all(not arg[0].isupper() for arg in self.arguments)

    def __repr__(self):
        return f"{self.predicate.name}({', '.join(self.arguments)})"

class Rule:
    """Represents a PSL rule (weighted first-order logic clause)."""
    def __init__(self, weight: float, body: List[Atom], head: List[Atom]):
        self.weight = weight
        self.body = body
        self.head = head

    def __repr__(self):
        body_str = " & ".join(map(str, self.body)) if self.body else "True"
        head_str = " | ".join(map(str, self.head)) if self.head else "False"
        return f"{self.weight}: {body_str} -> {head_str}"

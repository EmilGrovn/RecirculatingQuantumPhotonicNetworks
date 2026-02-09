#rqpnn_spec.py
#define system
from __future__ import annotations
from dataclasses import dataclass, field, asdict #for creating dataclass objects
from typing import Iterable, Tuple

@dataclass(frozen=True, slots=True)#Parts of this dataclass was written with AI - validated in test_rqpnn_spec
class RQPNNSpec:
    """
    Minimal, validated specification of the RQPNN system.

    Parameters
    ----------
    M_cav : int
        Number of cavities (must be a positive integer).
    M_em : int
        Number of emitters. Must be either 0 or equal to M_cav.
    N_exc_list : tuple[int, ...]
        Excitation-number sectors to include (non-negative integers in ascending order).
    include_checks : bool
        If True, enable extra consistency checks elsewhere in the codebase.
    """
    #initialize system parameters
    M_cav: int
    M_em: int
    N_exc_list: Tuple[int, ...] #we enforce N_exc_list to be a tuple for immutability
    include_checks: bool = False

    # Derive cached fields -> these are initialized here and will be set in __post_init__
    has_emitters: bool = field(init=False)

    #post-initialization: runs directly after __init__
    def __post_init__(self):
        # Basic type/value checks
        if not (isinstance(self.M_cav, int) and self.M_cav > 0):
            raise ValueError("M_cav must be a positive integer.")
        if self.M_em not in (0, self.M_cav):
            raise ValueError("M_em must be 0 or equal to M_cav.")
        if not isinstance(self.N_exc_list, Tuple):
            raise ValueError("N_exc_list must be a tuple")
        if len(self.N_exc_list) == 0:
            raise ValueError("N_exc_list must not be empty.")
        if any(n < 0 for n in self.N_exc_list) or not all(isinstance(n, int) for n in self.N_exc_list):
            raise ValueError("All elements of N_exc_list must be non-negative integers.")
        if not sorted(self.N_exc_list) == list(self.N_exc_list):
            raise ValueError("Elements of N_exc_list must be sorted in ascending order.")

        # Because the dataclass is frozen, use object.__setattr__ for derived/normalized fields
        object.__setattr__(self, "has_emitters", self.M_em == self.M_cav and self.M_em > 0)
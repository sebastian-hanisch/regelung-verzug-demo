"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))


@pytest.fixture
def mini():
    """Der gerundete Entwurf K = (2/5, 6/5) (nahe R = 1: K = (0,422; 1,244)) am Wagen A = [[1, 1], [0, 1]], B = [0, 1]ᵀ.
    Handrechnung: P_0 = z² − (4/5) z + 1/5 (Wurzeln 2/5 ± i/5), P_1 = z³ − 2 z² + (11/5) z − 4/5 (instabil)."""
    return F(2, 5), F(6, 5)

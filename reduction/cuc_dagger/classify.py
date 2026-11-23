"""Compatibility import: axial classification now uses Clifford double cosets."""
from direct_search.axial import z_rotation_metadata


def classify_axial(unitary, tol=1e-10):
    return z_rotation_metadata(unitary, tol)

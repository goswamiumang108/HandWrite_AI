import pytest
from handwrite_ai.logic import mp_neuron, gate


def test_logic_truth_tables():
    for a in (0, 1):
        assert gate("NOT", a) == (1 - a)
        for b in (0, 1):
            assert gate("AND", a, b) == (a & b)
            assert gate("OR", a, b) == (a | b)
            assert gate("NAND", a, b) == (1 - (a & b))


def test_invalid_neuron_inputs():
    with pytest.raises(ValueError):
        mp_neuron([0, 1], [1], 1)
    with pytest.raises(ValueError):
        mp_neuron([2], [1], 1)

import pytest
from rqpnn.rqpnn_spec import RQPNNSpec

def test_basic_spec():
    spec = RQPNNSpec(3, 0, (1,2,3))
    assert spec.has_emitters is False
    assert spec.N_exc_list == (1,2,3)

def test_emitters():
    spec = RQPNNSpec(4, 4, (1,))#Important to use comma for single-element tuple!
    assert spec.has_emitters is True

def test_invalid_emitter_number():
    with pytest.raises(ValueError):
        RQPNNSpec(4, 2, (1,2))

def test_invalid_M_cav_object():
    with pytest.raises(ValueError):
        RQPNNSpec(4.0, 0, (1,2))

def test_invalid_M_em_object():
    with pytest.raises(ValueError):
        RQPNNSpec(4, -1, (1,2))

def test_invalid_N_exc_list():
    with pytest.raises(ValueError):
        RQPNNSpec(4, 0, [1,2])
    with pytest.raises(ValueError):
        RQPNNSpec(4, 0, ())
    with pytest.raises(ValueError):
        RQPNNSpec(4, 0, (1,-2))
    with pytest.raises(ValueError):
        RQPNNSpec(4, 0, (2,1))
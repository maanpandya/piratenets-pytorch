import numpy as np

from piratenets.problems.allen_cahn import load_reference


def test_reference_shape_and_initial_condition():
    ref = load_reference()
    assert ref.usol.shape == (201, 512)
    assert ref.t.shape == (201,) and ref.x.shape == (512,)
    assert ref.t[0] == 0.0 and np.isclose(ref.t[-1], 1.0)
    # periodic grid x_j = -1 + 2 j / 512 (x = 1 is the same point as x = -1)
    np.testing.assert_allclose(ref.x, -1.0 + 2.0 * np.arange(512) / 512, atol=1e-15)
    np.testing.assert_allclose(ref.usol[0], ref.x**2 * np.cos(np.pi * ref.x), atol=1e-12)
    assert np.isfinite(ref.usol).all() and np.abs(ref.usol).max() <= 1.0 + 1e-8

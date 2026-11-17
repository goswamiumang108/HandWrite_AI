import numpy as np
from handwrite_ai.demo import train_demo, demo_predict


def test_offline_demo_is_genuinely_trained():
    model, accuracy, images, labels = train_demo()
    assert 0.85 < accuracy <= 1
    assert images.shape[1:] == (8, 8)
    p = demo_predict((images[0] / 16).astype("float32"))
    assert p.shape == (10,)
    np.testing.assert_allclose(p.sum(), 1, atol=1e-6)

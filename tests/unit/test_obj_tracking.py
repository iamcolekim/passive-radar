import os
import sys
import numpy as np

# Ensure project root is on sys.path for direct script runs
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from obj_tracking import CFARDetectorVectorized, KalmanTracker


def test_cfar_detects_single_target():
    rng = np.random.default_rng(0)
    heatmap = rng.exponential(scale=10.0, size=(128, 128))
    heatmap[64, 64] = 200.0

    detector = CFARDetectorVectorized(guard_cells=2, ref_cells=12, pfa=1e-4)
    detections, mask = detector.detect(heatmap)

    assert mask[64, 64] is True
    assert any((d == [64, 64]).all() for d in detections)


def test_cfar_detections_to_physics_outputs_units():
    detector = CFARDetectorVectorized()
    detections = np.array([[10, 20], [15, 30]])

    results = detector.detections_to_physics(
        detections,
        fs=2_000_000.0,
        N=128,
        fc=509_000_000.0,
        delay_0_idx=0,
        doppler_0_idx=64,
    )

    assert len(results) == 2
    for item in results:
        assert "bistatic_range_m" in item
        assert "doppler_hz" in item
        assert "bistatic_velocity_ms" in item


def test_kalman_converges_on_constant_velocity():
    tracker = KalmanTracker(dt=1.0)
    tracker.x = np.array([0.0, 1.0, 0.0, 2.0])

    true = tracker.x.copy()
    for _ in range(20):
        true[0] += true[1]
        true[2] += true[3]
        meas = [true[0], true[2]]
        tracker.predict()
        est = tracker.update(meas)

    assert abs(est[0] - true[0]) < 1.0
    assert abs(est[2] - true[2]) < 1.0

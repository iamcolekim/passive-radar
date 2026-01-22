from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

class CFARDetectorVectorized:
    """
    Vectorized 2D CA-CFAR detector.

    Units / I-O Contract:
    - Input heatmap is **power** (magnitude-squared), non-negative.
    - Output detections are [row, col] indices (bin space).
    """

    def __init__(self, guard_cells: int = 2, ref_cells: int = 10, pfa: float = 1e-6):
        self.guard = guard_cells
        self.ref = ref_cells
        self.pfa = pfa
        
        # 1. Calculate the Threshold Factor (Alpha)
        # N = Total number of training cells in the "donut"
        # Total Area = (2R + 2G + 1)^2
        # Guard Area = (2G + 1)^2
        # N = Total Area - Guard Area
        window_dim = 2 * (ref_cells + guard_cells) + 1
        guard_dim = 2 * guard_cells + 1
        self.num_train_cells = (window_dim**2) - (guard_dim**2)
        
        # Standard CA-CFAR alpha for square-law detector
        self.alpha = self.num_train_cells * (pfa**(-1/self.num_train_cells) - 1)

        # 2. Pre-build the Convolution Kernel (The "Donut")
        self.kernel = np.ones((window_dim, window_dim))
        
        # Zero out the center (Guard cells + CUT)
        # The center of the matrix is at index [ref + guard]
        center_idx = ref_cells + guard_cells
        start_guard = center_idx - guard_cells
        end_guard = center_idx + guard_cells + 1
        
        self.kernel[start_guard:end_guard, start_guard:end_guard] = 0
        
        # Normalize the kernel so convolution calculates the MEAN immediately
        self.kernel = self.kernel / self.num_train_cells

    def detect(self, heatmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            heatmap: 2D numpy array (Range-Doppler map) in **power**.

        Returns:
            detections: Nx2 array of [row, col] indices.
            detection_mask: boolean array (True = detection).
        """
        # 1. Estimate Noise Floor everywhere simultaneously
        # convolve 'smears' the map with our donut kernel to get local averages
        noise_floor = scipy.ndimage.convolve(heatmap, self.kernel, mode='constant', cval=np.mean(heatmap))
        
        # 2. Calculate Threshold Map
        threshold_map = noise_floor * self.alpha
        
        # 3. Apply Threshold (Vectorized comparison)
        # We perform a boolean check on the entire matrix at once
        detection_mask = heatmap > threshold_map
        
        # 4. Extract Coordinates
        # argwhere returns (row, col) for every True value
        detections = np.argwhere(detection_mask)
        
        return detections, detection_mask

    def detections_to_physics(
        self,
        detections: np.ndarray,
        fs: float,
        N: int,
        fc: float,
        delay_0_idx: float = 0,
        doppler_0_idx: float | None = None,
    ) -> List[Dict[str, float]]:
        """
        Convert CFAR detections (bin indices) to SI units.

        Args:
            detections: Nx2 array of [row, col] indices.
            fs: Sample rate (Hz).
            N: Doppler FFT size (bins).
            fc: Carrier frequency (Hz).
            delay_0_idx: Row index corresponding to 0 delay.
            doppler_0_idx: Col index corresponding to 0 Doppler.

        Returns:
            List of dicts with range/doppler in both bin and SI units.
        """
        if doppler_0_idx is None:
            doppler_0_idx = N / 2

        c = 299_792_458.0
        results: List[Dict[str, float]] = []
        for r_bin, d_bin in np.asarray(detections):
            delay_samples = float(r_bin) - delay_0_idx
            range_m = (delay_samples / fs) * c
            doppler_shift_bins = float(d_bin) - doppler_0_idx
            doppler_hz = doppler_shift_bins * (fs / N)
            wavelength = c / fc
            velocity_ms = (doppler_hz * wavelength) / 2
            results.append({
                "range_bin": float(r_bin),
                "doppler_bin": float(d_bin),
                "bistatic_range_m": range_m,
                "doppler_hz": doppler_hz,
                "bistatic_velocity_ms": velocity_ms,
            })
        return results

class KalmanTracker:
    """
    Standard Linear Kalman Filter for Constant Velocity Model.

    State Vector x = [range_bin, range_rate_bin, doppler_bin, doppler_rate_bin]^T
    Units / I-O Contract:
    - State and measurements are in **bin units**.
    - Use `get_physics_state()` to convert to SI units.
    """

    def __init__(self, dt: float = 0.5):
        self.dt = dt
        self.x = np.zeros(4) 
        
        # State Transition (Physics: Position += Velocity * dt)
        self.F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        
        # Measurement Matrix (We observe Range and Doppler only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        
        # COVARIANCE MATRICES (The "Tuning" Knobs)
        # P: Initial uncertainty (We are very unsure at start)
        self.P = np.eye(4) * 1000  
        
        # R: Measurement Noise Covariance (Sensor accuracy)
        # How much do we trust the CAF peak location? 
        # Range err ~10m, Doppler err ~1Hz
        self.R = np.diag([10**2, 1**2]) 
        
        # Q: Process Noise Covariance (Real world jitters)
        # How much does the target jerk around? (Low for planes, high for drones)
        self.Q = np.eye(4) * 0.1   

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def get_physics_state(
        self,
        fs: float,
        N: int,
        fc: float,
        delay_0_idx: float = 0,
        doppler_0_idx: float | None = None,
    ) -> Dict[str, float]:
        """
        Converts internal bin-tracking state to SI units.
        
        Parameters:
        - fs: Sample Rate (Hz)
        - N: Block Size (Number of Doppler bins)
        - fc: Carrier Frequency (Hz) (e.g., 509e6)
        - delay_0_idx: Which row index corresponds to 0 delay? (Usually 0)
        - doppler_0_idx: Which col index corresponds to 0 Hz? (Usually N/2 for centered plots)
        
        Returns:
        - dict with 'bistatic_range_m', 'doppler_hz', 'velocity_m_s'
        """
        c = 299_792_458 # Speed of light (m/s)
        
        # 1. Extract raw bin positions from state vector
        r_bin = self.x[0]
        d_bin = self.x[2]
        
        # 2. Convert Range Bin -> Bistatic Range (Meters)
        # Time Delay = (Bin - 0_offset) / fs
        # Distance = Time * c
        delay_samples = r_bin - delay_0_idx
        range_m = (delay_samples / fs) * c
        
        # 3. Convert Doppler Bin -> Frequency Shift (Hz)
        # Resolution = fs / N
        if doppler_0_idx is None:
            doppler_0_idx = N / 2  # Assume centered convention
            
        doppler_shift_bins = d_bin - doppler_0_idx
        doppler_hz = doppler_shift_bins * (fs / N)
        
        # 4. Convert Doppler (Hz) -> Bistatic Velocity (m/s)
        # Standard Radar Equation: fd = (2 * v) / lambda
        # Therefore: v = (fd * lambda) / 2
        wavelength = c / fc
        velocity_ms = (doppler_hz * wavelength) / 2
        
        return {
            "range_bin": r_bin,
            "doppler_bin": d_bin,
            "bistatic_range_m": range_m,
            "doppler_hz": doppler_hz,
            "bistatic_velocity_ms": velocity_ms
        }

    def update(self, measurement: Iterable[float]) -> np.ndarray:
        # measurement = [r_meas, d_meas]
        z = np.array(measurement)
        
        # Innovation (Residual)
        y = z - self.H @ self.x 
        
        # Innovation Covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman Gain (Optimal Weighting)
        K = self.P @ self.H.T @ np.linalg.inv(S) 
        
        # State Update
        self.x = self.x + K @ y
        
        # Covariance Update
        I = np.eye(self.F.shape[0])
        self.P = (I - K @ self.H) @ self.P
        
        return self.x

# --- SIMULATION DEMO ---
if __name__ == "__main__":
    # 1. Setup
    # Create a 200x200 dummy map with noise
    heatmap_dim = 200
    noise_power = 10.0
    heatmap = np.random.exponential(scale=noise_power, size=(heatmap_dim, heatmap_dim))
    
    # 2. Inject a "Target" (Plane)
    # Target is brighter than noise (SNR ~ 15dB)
    target_row, target_col = 50, 50
    heatmap[target_row, target_col] = 150.0 
    
    # 3. Run Fast CFAR
    print("Running Vectorized CFAR...")
    detector = CFARDetectorVectorized(guard_cells=2, ref_cells=15, pfa=1e-4)
    detections, mask = detector.detect(heatmap)
    
    print(f"Detected {len(detections)} potential targets.")
    if len(detections) > 0:
        print(f"First detection at indices: {detections[0]}")
    
    # 4. Run Tracker on the detected peak
    tracker = KalmanTracker(dt=0.5)
    # Initialize tracker near the first detection for demo purposes
    tracker.x = np.array([target_row, 2, target_col, 0.5]) 
    
    print("\n--- Tracking Simulation ---")
    print(f"{'Step':<5} {'Meas Range':<12} {'Meas Dopp':<12} {'Est Range':<12} {'Est Dopp':<12}")
    
    # Simulate the target moving for 10 steps
    true_r, true_d = target_row, target_col
    vel_r, vel_d = 2.0, 0.5
    
    for t in range(10):
        # Move "Truth"
        true_r += vel_r
        true_d += vel_d
        
        # Predict
        tracker.predict()
        
        # Measure (Add noise to simulate CAF jitter)
        meas_r = true_r + np.random.normal(0, 1.0)
        meas_d = true_d + np.random.normal(0, 0.5)
        
        # Update
        est = tracker.update([meas_r, meas_d])
        
        print(f"{t:<5} {meas_r:<12.2f} {meas_d:<12.2f} {est[0]:<12.2f} {est[2]:<12.2f}")
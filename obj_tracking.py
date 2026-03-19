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
        """
        noise_floor = scipy.ndimage.convolve(heatmap, self.kernel, mode='constant', cval=np.mean(heatmap))
        # Note for above: borders may be locally less than global mean, possible failure modes if targets are near edges.
        # Suggesting exclude edge CUTs (Cells Under Test), mirror/reflect, circular/wrap, or trimmed windows (partial CFAR) near edges to maintain local context (some may still introduce artifacts).
        """
        # Safer implementation:
        noise_floor = scipy.ndimage.convolve(heatmap, self.kernel, mode='mirror')
        
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
        Tblock: float, 
        delay_0_idx: float = 0,
        doppler_0_idx: float | None = None,
    ) -> Dict[str, float]:
        c = 299_792_458.0
        
        r_bin = self.x[0]
        r_dot_bin = self.x[1]  # <--- Range Rate (bins per Tblock)
        d_bin = self.x[2]
        d_dot_bin = self.x[3]  # <--- Doppler Rate (bins per Tblock)
        
        # Position Math
        delay_samples = r_bin - delay_0_idx
        range_m = ((delay_samples / fs) * c) / 2.0 
        
        if doppler_0_idx is None:
            doppler_0_idx = N / 2
        doppler_hz = (d_bin - doppler_0_idx) * (1.0 / (N * Tblock))
        
        # Velocity Math (Time-Derivatives)
        # Convert bins/Tblock to SI/second
        range_rate_m_s = ((r_dot_bin / fs * c) / 2.0) / Tblock
        doppler_rate_hz_s = (d_dot_bin * (1.0 / (N * Tblock))) / Tblock
        
        bistatic_velocity_ms = (doppler_hz * (c / fc)) / 2.0
        
        return {
            "range_bin": r_bin,
            "doppler_bin": d_bin,
            "bistatic_range_m": range_m,
            "doppler_hz": doppler_hz,
            "bistatic_velocity_ms": bistatic_velocity_ms,
            "range_rate_m_s": range_rate_m_s,       
            "doppler_rate_hz_s": doppler_rate_hz_s  
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
    
class SimpleTracker:
    """Lean single-target tracker for passive radar CAF frames."""
    
    def __init__(self, fs: float, N: int, fc: float, Tblock: float, 
                 pfa: float = 1e-5, guard_cells: int = 2, ref_cells: int = 8):
        self.fs = fs
        self.N = N
        self.fc = fc
        self.Tblock = Tblock
        
        self.detector = CFARDetectorVectorized(
            guard_cells=guard_cells, ref_cells=ref_cells, pfa=pfa
        )
        self.tracker: KalmanTracker | None = None
        
    def process_caf_frame(self, caf_power: np.ndarray, frame_idx: int, doppler_0_idx: int) -> dict:
        detections, _ = self.detector.detect(caf_power)
        
        result = {'frame': frame_idx, 'active': False, 'r_bin': None, 'd_bin': None}
        
        if self.tracker is None and len(detections) > 0:
            det_powers = [caf_power[int(d[0]), int(d[1])] for d in detections]
            strongest_idx = np.argmax(det_powers)
            r_bin, d_bin = detections[strongest_idx]
            
            self.tracker = KalmanTracker(dt=self.Tblock)
            self.tracker.x = np.array([r_bin, 0.0, d_bin, 0.0])
            print(f"  → Tracker locked at frame {frame_idx}")
            
        if self.tracker is not None:
            predicted = self.tracker.predict()
            
            if len(detections) > 0:
                pred_pos = np.array([predicted[0], predicted[2]])
                distances = [np.linalg.norm(d - pred_pos) for d in detections]
                closest_idx = np.argmin(distances)
                
                if distances[closest_idx] < 15:  
                    self.tracker.update(detections[closest_idx])
                    
            result['active'] = True
            result['state'] = self.tracker.get_physics_state(
                fs=self.fs, 
                N=self.N, 
                fc=self.fc, 
                Tblock=self.Tblock, # <--- Pass Tblock down
                delay_0_idx=0, 
                doppler_0_idx=doppler_0_idx # <--- Pass the dynamic zero index
            )
            
        # Add the raw detections to the output!
        result['raw_detections'] = detections 
        return result


class MultiTargetTracker:
    """
    Lean Multi-Target Tracker for urban passive radar scenarios.
    Handles Data Association, Coasting, and Track Lifecycles without hoarding history.
    """
    
    def __init__(self, fs: float, N: int, fc: float, Tblock: float,
                 pfa: float = 1e-6, max_coast: int = 5,
                 association_threshold: float = 20.0):
        self.fs = fs
        self.N = N
        self.fc = fc
        self.Tblock = Tblock
        
        # Detector
        self.detector = CFARDetectorVectorized(guard_cells=2, ref_cells=10, pfa=pfa)
        
        # Track Management
        self.tracks: List[KalmanTracker] = []
        self.track_ids: List[int] = []
        self.coast_counters: List[int] = []
        
        self.max_coast = max_coast
        self.association_threshold = association_threshold
        self.next_track_id = 0
        
    def process_caf_frame(self, caf_power: np.ndarray, frame_idx: int,
                         delay_0_idx: float = 0, doppler_0_idx: float | None = None) -> Dict:
        if doppler_0_idx is None:
            doppler_0_idx = self.N / 2
            
        detections, _ = self.detector.detect(caf_power)
        
        # 1. Predict all existing tracks
        for tracker in self.tracks:
            tracker.predict()
        
        # 2. Data Association (Nearest Neighbor)
        associated = self._associate_detections(detections)
        
        dropped_tracks = []
        
        # 3. Update & Coast
        for i in range(len(self.tracks)):
            if associated[i] is not None:
                self.tracks[i].update(associated[i])
                self.coast_counters[i] = 0  # Reset coasting if hit
            else:
                self.coast_counters[i] += 1 # Coast if missed
                
                # Tag for dropping if coasting too long
                if self.coast_counters[i] > self.max_coast:
                    dropped_tracks.append(self.track_ids[i])
        
        # 4. Remove Dropped Tracks
        for track_id in dropped_tracks:
            idx = self.track_ids.index(track_id)
            del self.tracks[idx]
            del self.track_ids[idx]
            del self.coast_counters[idx]
            print(f"  [-] Dropped track {track_id}")
        
        # 5. Spawn New Tracks for Unassociated Detections
        # Find which detections were claimed by existing tracks
        used_det_indices = {i for i, assoc in enumerate(associated) if assoc is not None}
        
        for det_idx, det in enumerate(detections):
            if det_idx not in used_det_indices:
                new_tracker = KalmanTracker(dt=self.Tblock)
                new_tracker.x = np.array([det[0], 0.0, det[1], 0.0])
                
                self.tracks.append(new_tracker)
                self.track_ids.append(self.next_track_id)
                self.coast_counters.append(0)
                
                print(f"  [+] Spawned track {self.next_track_id} at Frame {frame_idx}")
                self.next_track_id += 1
        
        # 6. Extract Physics States for Plotting
        current_tracks = []
        for tracker, track_id in zip(self.tracks, self.track_ids):
            state_si = tracker.get_physics_state(
                fs=self.fs, N=self.N, fc=self.fc, Tblock=self.Tblock, 
                delay_0_idx=delay_0_idx, doppler_0_idx=doppler_0_idx
            )
            current_tracks.append({'track_id': track_id, **state_si})
        
        return {
            'frame': frame_idx,
            'tracks': current_tracks,
            'raw_detections': detections # Returned so your CFAR shotgun plot still works!
        }
    
    def _associate_detections(self, detections: np.ndarray) -> List[np.ndarray | None]:
        """Nearest neighbor data association logic."""
        associated = [None] * len(self.tracks)
        used_detections = set()
        
        for i, tracker in enumerate(self.tracks):
            predicted_pos = np.array([tracker.x[0], tracker.x[2]])
            
            best_dist = float('inf')
            best_det_idx = None
            
            for j, det in enumerate(detections):
                if j in used_detections:
                    continue
                
                dist = np.linalg.norm(det - predicted_pos)
                if dist < best_dist and dist < self.association_threshold:
                    best_dist = dist
                    best_det_idx = j
            
            if best_det_idx is not None:
                associated[i] = detections[best_det_idx]
                used_detections.add(best_det_idx)
        
        return associated
    
# --- SIMULATION DEMO ---
if __name__ == "__main__":
    # 1. Setup
    # Create a 200x200 dummy map with noise
    heatmap_dim = 200
    noise_power = 10.0
    rng = np.random.default_rng(0)
    heatmap = rng.exponential(scale=noise_power, size=(heatmap_dim, heatmap_dim))
    
    # 2. Inject a "Target" (Plane)
    # Target is brighter than noise (SNR ~ 15dB)
    signal_power = 150.0
    target_row, target_col = 50, 50
    heatmap[target_row, target_col] = signal_power 
    
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
    
    # Calculate SNR
    noise_mean = np.mean(heatmap)
    snr_linear = signal_power / noise_mean
    snr_db = 10 * np.log10(snr_linear) # power, so 10*log10, not 20*log10

    
    print("\n--- Tracking Simulation ---")
    print(f"Signal Power: {signal_power:.2f}, Noise Mean: {noise_mean:.2f}, SNR: {snr_db:.2f} dB\n")
    print(f"{'Step':<5} {'True R':<10} {'True D':<10} {'Meas R':<10} {'Meas D':<10} {'Est R':<10} {'Est D':<10} {'MSE T-E':<12} {'MSE M-E':<12}")
    print("-" * 20)
    
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
        meas_r = true_r + rng.normal(0, 1.0)
        meas_d = true_d + rng.normal(0, 0.5)
        
        # Update
        est = tracker.update([meas_r, meas_d])
        
        # Calculate MSE: True vs Estimated
        mse_true_est = ((true_r - est[0])**2 + (true_d - est[2])**2) / 2
        
        # Calculate MSE: Measured vs Estimated
        mse_meas_est = ((meas_r - est[0])**2 + (meas_d - est[2])**2) / 2
        
        print(f"{t:<5} {true_r:<10.2f} {true_d:<10.2f} {meas_r:<10.2f} {meas_d:<10.2f} {est[0]:<10.2f} {est[2]:<10.2f} {mse_true_est:<12.4f} {mse_meas_est:<12.4f}")

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
    
class SimpleTracker:
    """
    Lightweight single-target tracker for passive radar CAF frames.
    Combines CFAR detection with Kalman filtering.
    """
    
    def __init__(self, fs: float, N: int, fc: float, Tblock: float, 
                 pfa: float = 1e-5, guard_cells: int = 2, ref_cells: int = 8, enable_mse: bool = False):
        """
        Args:
            fs: Sample rate (Hz)
            N: Doppler FFT size (number of bins)
            fc: Carrier frequency (Hz)
            Tblock: Time between frames (seconds)
            pfa: Probability of false alarm for CFAR
            guard_cells: CFAR guard cells
            ref_cells: CFAR reference cells
            enable_mse: When False, skip all MSE computation/storage.
        """
        self.fs = fs
        self.N = N
        self.fc = fc
        self.Tblock = Tblock
        
        # CFAR detector
        self.detector = CFARDetectorVectorized(
            guard_cells=guard_cells, 
            ref_cells=ref_cells, 
            pfa=pfa
        )
        
        # Single target tracker (initialize on first detection)
        self.tracker: KalmanTracker | None = None
        self.track_history: List[Dict[str, float]] = []
        
        self.mse_history: List[float] = []
        self.enable_mse = enable_mse
        
    def process_caf_frame(self, caf_mag_db: np.ndarray, frame_idx: int, 
                         delay_0_idx: float = 0, doppler_0_idx: float | None = None) -> Dict:
        """
        Process one CAF frame with detection and tracking.
        
        Args:
            caf_mag_db: 2D array (range x doppler) in dB magnitude
            frame_idx: Current frame number
            delay_0_idx: Row index for zero delay
            doppler_0_idx: Column index for zero Doppler (defaults to N/2)
            
        Returns:
            dict with keys:
                - 'frame': frame index
                - 'num_detections': number of CFAR detections
                - 'detections': Nx2 array of [row, col] bin indices
                - 'det_mask': boolean detection mask
                - 'track_state': dict with SI units (None if no track)
                - 'mse': mean-squared error of prediction (None if disabled/no update)
        """
        if doppler_0_idx is None:
            doppler_0_idx = self.N / 2
        
        # Convert dB magnitude to power for CFAR
        caf_power = 10 ** (caf_mag_db / 10)
        
        # Run CFAR detection
        detections, det_mask = self.detector.detect(caf_power)
        
        result = {
            'frame': frame_idx,
            'num_detections': len(detections),
            'detections': detections,
            'det_mask': det_mask,
            'track_state': None,
            'mse': None
        }
        
        # Initialize tracker on first strong detection
        if self.tracker is None and len(detections) > 0:
            # Pick strongest detection as initial target
            det_powers = [caf_power[int(d[0]), int(d[1])] for d in detections]
            strongest_idx = np.argmax(det_powers)
            r_bin, d_bin = detections[strongest_idx]
            
            self.tracker = KalmanTracker(dt=self.Tblock)
            # Initialize state: [range_bin, range_rate, doppler_bin, doppler_rate]
            self.tracker.x = np.array([r_bin, 0.0, d_bin, 0.0])
            
            print(f"  → Tracker initialized at frame {frame_idx}: "
                  f"R={r_bin:.1f} bins, D={d_bin:.1f} bins")
        
        # Track existing target
        if self.tracker is not None:
            # Predict next state
            predicted = self.tracker.predict()
            
            # Find closest detection to prediction (data association)
            if len(detections) > 0:
                pred_pos = np.array([predicted[0], predicted[2]])  # [range_bin, doppler_bin]
                distances = [np.linalg.norm(d - pred_pos) for d in detections]
                closest_idx = np.argmin(distances)
                
                # Update if close enough (gating threshold)
                if distances[closest_idx] < 15:  # bins
                    measurement = detections[closest_idx]
                    self.tracker.update(measurement)
                    
                    if self.enable_mse:
                        innovation = measurement - pred_pos
                        mse = float(np.mean(innovation ** 2))
                        self.mse_history.append(mse)
                        result['mse'] = mse
            
            # Get current state in SI units
            state_si = self.tracker.get_physics_state(
                self.fs, self.N, self.fc, delay_0_idx, doppler_0_idx
            )
            self.track_history.append(state_si)
            result['track_state'] = state_si
        
        return result
    
    def get_track_summary(self) -> Dict[str, np.ndarray]:
        """
        Get arrays of tracked quantities over time.
        
        Returns:
            dict with 'ranges_m', 'velocities_ms', 'mse', 'frames', 'mse_available'
        """
        if not self.track_history:
            return {
                'ranges_m': np.array([]),
                'velocities_ms': np.array([]),
                'range_bins': np.array([]),
                'doppler_bins': np.array([]),
                'mse': np.array(self.mse_history if self.enable_mse else []),
                'mse_available': self.enable_mse,
                'frames': np.array([])
            }
        
        return {
            'ranges_m': np.array([s['bistatic_range_m'] for s in self.track_history]),
            'velocities_ms': np.array([s['bistatic_velocity_ms'] for s in self.track_history]),
            'range_bins': np.array([s['range_bin'] for s in self.track_history]),
            'doppler_bins': np.array([s['doppler_bin'] for s in self.track_history]),
            'mse': np.array(self.mse_history if self.enable_mse else []),
            'mse_available': self.enable_mse,
            'frames': np.arange(len(self.track_history))
        }


class MultiTargetTracker: # do not use yet as an import, still in development
    """
    Multi-target tracker with track initialization, association, and termination.
    For more complex scenarios with multiple aircraft.
    """
    
    def __init__(self, fs: float, N: int, fc: float, Tblock: float,
                 pfa: float = 1e-5, max_coast: int = 5,
                 association_threshold: float = 20.0, enable_mse: bool = False):
        """
        Args:
            fs: Sample rate (Hz)
            N: Doppler FFT size
            fc: Carrier frequency (Hz)
            Tblock: Time between frames (seconds)
            pfa: CFAR probability of false alarm
            max_coast: Max frames to coast without detection before dropping track
            association_threshold: Max distance (bins) for measurement association
            enable_mse: When False, skip all MSE computation/storage.
        """
        self.fs = fs
        self.N = N
        self.fc = fc
        self.Tblock = Tblock
        
        self.detector = CFARDetectorVectorized(guard_cells=2, ref_cells=10, pfa=pfa)
        
        # Track management
        self.tracks: List[KalmanTracker] = []
        self.track_ids: List[int] = []
        self.coast_counters: List[int] = []
        self.max_coast = max_coast
        self.association_threshold = association_threshold
        self.enable_mse = enable_mse
        self.next_track_id = 0
        
        # History
        self.track_history: Dict[int, List[Dict[str, float]]] = {}
        self.mse_history: Dict[int, List[float]] = {}
        
    def process_caf_frame(self, caf_mag_db: np.ndarray, frame_idx: int,
                         delay_0_idx: float = 0, doppler_0_idx: float | None = None) -> Dict:
        """
        Process one frame with multi-target tracking.
        
        Returns:
            dict with 'detections', 'tracks', 'new_tracks', 'dropped_tracks'
        """
        if doppler_0_idx is None:
            doppler_0_idx = self.N / 2
        
        caf_power = 10 ** (caf_mag_db / 10)
        detections, det_mask = self.detector.detect(caf_power)
        
        # Predict all tracks
        for tracker in self.tracks:
            tracker.predict()
        
        # Data association (nearest neighbor)
        associated = self._associate_detections(detections)
        
        new_tracks = []
        dropped_tracks = []
        frame_mse_by_track: Dict[int, float] = {}
        
        # Update tracks
        for i in range(len(self.tracks)):
            if associated[i] is not None:
                pred_pos = np.array([self.tracks[i].x[0], self.tracks[i].x[2]], dtype=np.float64)
                self.tracks[i].update(associated[i])
                self.coast_counters[i] = 0

                if self.enable_mse:
                    innovation = associated[i] - pred_pos
                    mse = float(np.mean(innovation ** 2))
                    track_id = self.track_ids[i]
                    frame_mse_by_track[track_id] = mse
                    if track_id not in self.mse_history:
                        self.mse_history[track_id] = []
                    self.mse_history[track_id].append(mse)
                
                # Store history
                state_si = self.tracks[i].get_physics_state(
                    self.fs, self.N, self.fc, delay_0_idx, doppler_0_idx
                )
                track_id = self.track_ids[i]
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append(state_si)
            else:
                self.coast_counters[i] += 1
                
                if self.coast_counters[i] > self.max_coast:
                    dropped_tracks.append(self.track_ids[i])
        
        # Remove dropped tracks
        for track_id in dropped_tracks:
            idx = self.track_ids.index(track_id)
            del self.tracks[idx]
            del self.track_ids[idx]
            del self.coast_counters[idx]
        
        # Initialize new tracks from unassociated detections
        used_det_indices = {i for i, assoc in enumerate(associated) if assoc is not None}
        for det_idx, det in enumerate(detections):
            if det_idx not in used_det_indices:
                new_tracker = KalmanTracker(dt=self.Tblock)
                new_tracker.x = np.array([det[0], 0.0, det[1], 0.0])
                
                self.tracks.append(new_tracker)
                self.track_ids.append(self.next_track_id)
                self.coast_counters.append(0)
                new_tracks.append(self.next_track_id)
                self.track_history[self.next_track_id] = []
                if self.enable_mse:
                    self.mse_history[self.next_track_id] = []
                
                self.next_track_id += 1
        
        # Current track states
        current_tracks = []
        for tracker, track_id in zip(self.tracks, self.track_ids):
            state_si = tracker.get_physics_state(
                self.fs, self.N, self.fc, delay_0_idx, doppler_0_idx
            )
            current_tracks.append({'track_id': track_id, **state_si})
        
        return {
            'frame': frame_idx,
            'detections': detections,
            'tracks': current_tracks,
            'new_tracks': new_tracks,
            'dropped_tracks': dropped_tracks,
            'det_mask': det_mask,
            'mse_by_track': frame_mse_by_track if self.enable_mse else None,
            'mse_available': self.enable_mse,
        }

    def get_track_summary(self, track_id: int | None = None) -> Dict:
        """
        Get summary arrays for one track or all tracks.

        Args:
            track_id: specific track to summarize, or None for all tracks.
        """
        if track_id is not None:
            hist = self.track_history.get(track_id, [])
            return {
                'track_id': track_id,
                'ranges_m': np.array([s['bistatic_range_m'] for s in hist]),
                'velocities_ms': np.array([s['bistatic_velocity_ms'] for s in hist]),
                'range_bins': np.array([s['range_bin'] for s in hist]),
                'doppler_bins': np.array([s['doppler_bin'] for s in hist]),
                'mse': np.array(self.mse_history.get(track_id, []) if self.enable_mse else []),
                'mse_available': self.enable_mse,
                'frames': np.arange(len(hist)),
            }

        return {
            'tracks': {
                tid: {
                    'ranges_m': np.array([s['bistatic_range_m'] for s in self.track_history.get(tid, [])]),
                    'velocities_ms': np.array([s['bistatic_velocity_ms'] for s in self.track_history.get(tid, [])]),
                    'range_bins': np.array([s['range_bin'] for s in self.track_history.get(tid, [])]),
                    'doppler_bins': np.array([s['doppler_bin'] for s in self.track_history.get(tid, [])]),
                    'mse': np.array(self.mse_history.get(tid, []) if self.enable_mse else []),
                    'frames': np.arange(len(self.track_history.get(tid, []))),
                }
                for tid in self.track_history.keys()
            },
            'mse_available': self.enable_mse,
        }
    
    def _associate_detections(self, detections: np.ndarray) -> List[np.ndarray | None]:
        """Nearest neighbor data association."""
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

import numpy as np
from gnuradio import gr

# Export for gnu-radio companion
class lms_canceller(gr.sync_block):
    """
    A GRC block for a sample-by-sample LMS adaptive filter.
    This is a sync_block because it is iterative, not block-based.
    """
    def __init__(self, filter_taps=64, step_size_mu=0.01):
        gr.sync_block.__init__(
            self,
            name='LMS Canceller',
            in_sig=[np.complex64, np.complex64],  # [ref (x), surv (d)]
            out_sig=[np.complex64]                # [error (e)]
        )
        
        self.K = int(filter_taps)
        self.mu = float(step_size_mu)
        
        # Initialize filter weights (w) and tap-delay buffer (x_buffer)
        self.w = np.zeros(self.K, dtype=np.complex64)
        self.x_buffer = np.zeros(self.K, dtype=np.complex64)

    def work(self, input_items, output_items):
        """
        This is called for every chunk of samples.
        """
        ref_in = input_items[0]   # Reference signal, x[n]
        surv_in = input_items[1]  # Surveillance signal (desired), d[n]
        out = output_items[0]     # Error signal (output), e[n]
        
        # Iterate sample-by-sample
        for i in range(len(ref_in)):
            # Update the reference signal buffer
            # (shift old samples, add new one)
            self.x_buffer[1:] = self.x_buffer[:-1]
            self.x_buffer[0] = ref_in[i]
            
            # 1. Calculate the filter output (y)
            # y = w^H * x
            y = np.dot(np.conj(self.w), self.x_buffer)
            
            # 2. Calculate the error signal (e)
            # e = d - y
            e = surv_in[i] - y
            
            # 3. Update the filter weights (w)
            # w = w + mu * e * x*
            self.w = self.w + self.mu * e * np.conj(self.x_buffer)
            
            # Set the output
            out[i] = e

        return len(out)
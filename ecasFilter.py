import numpy as np
from gnuradio import gr
import clutterCancellation # module header, used to load algorithm implementation

#Export for gnu-radio companion
class eca_canceller(gr.basic_block):
    """
    GRC wrapper for the clutterCancellation.ECAS function.
    This is a block-based algorithm, so we use gr.basic_block.
    """
    def __init__(self, filter_taps_K=10, max_doppler_D=3, num_batches_T=4, window_ext_Na=100, vector_size=32768):
        gr.basic_block.__init__(
            self,
            name='ECA-S Canceller',
            in_sig=[np.complex64, np.complex64],  # [ref, surv]
            out_sig=[np.complex64]                # [filtered_surv]
        )
        
        # Store our parameters
        self.K = int(filter_taps_K)
        self.D = int(max_doppler_D)
        self.T = int(num_batches_T)
        self.Na = int(window_ext_Na)
        self.N = int(vector_size)
        
        # Generate the subspace list ONCE during initialization
        self.subspace_list = clutterCancellation.gen_subspace_indexes(self.K, self.D)
        
        # Tell the scheduler we need N samples to produce N samples
        self.set_fixed_input_size(0, self.N) # Ref input
        self.set_fixed_input_size(1, self.N) # Surv input
        self.set_fixed_output_size(0, self.N) # Filtered output


    def general_work(self, input_items, output_items):
        """
        This is called when we have N samples in both inputs.
        """
        ref_ch = input_items[0]
        surv_ch = input_items[1]
        
        try:
            # Call the ECAS function from your .py file!
            filt_ch = clutterCancellation.ECAS(
                ref_ch, 
                surv_ch, 
                self.subspace_list,
                self.T,
                self.Na
            )
            
            # Copy the filtered data to the output buffer
            output_items[0][:] = filt_ch

        except Exception as e:
            print(f"Error in ECAS: {e}")
            output_items[0][:] = surv_ch # Pass through unfiltered
            
        return len(output_items[0])
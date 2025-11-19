import numpy as np
from gnuradio import gr
import clutterCancellation # module header, used to load algorithm implementation

#Export for gnu-radio companion
class wiener_canceller(gr.basic_block):
    """
    GRC wrapper for the clutterCancellation.Wiener_SMI function.

    This block is a 'basic_block', meaning it handles
    block-based (matrix) processing instead of sample-by-sample.
    """
    def __init__(self, filter_taps=64, vector_size=32768):
        gr.basic_block.__init__(
            self,
            name='Wiener SMI Canceller',
            in_sig=[np.complex64, np.complex64],  # [ref, surv]
            out_sig=[np.complex64]                # [filtered_surv]
        )
        
        # Store our parameters
        self.K = int(filter_taps)
        self.N = int(vector_size)
        
        # Tell the scheduler how many items we need to produce one output block
        # We process N samples at a time.
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
            # Call the function from your .py file!
            # We use the 'direct_matrix' implementation as it's faster
            # and GRC's block-based flow handles the memory.
            filt_ch, w = clutterCancellation.Wiener_SMI(
                ref_ch, 
                surv_ch, 
                self.K, 
                imp="direct_matrix"
            )
            
            # Copy the filtered data to the output buffer
            output_items[0][:] = filt_ch

        except Exception as e:
            print(f"Error in Wiener_SMI: {e}")
            # On error, just pass through the surveillance channel
            output_items[0][:] = surv_ch
            
        # Tell the scheduler how many items we produced
        # This will be N, but 'len(output_items[0])' is safer
        return len(output_items[0])
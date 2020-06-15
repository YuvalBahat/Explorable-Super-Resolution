# Consistency Enforcing Module (CEM)

An architectural module (implemented in PyTorch) that can enforce outputs consistency by wrapping **any super-resolution model (pre-trained or not):** Outputs of this module are guaranteed to match the low-resolution inputs, when downsampled using the given (or otherwise the bicubic) downsampling kernel.
<p align="center">
   <img src="fig_CEM_arch.png">
</p>

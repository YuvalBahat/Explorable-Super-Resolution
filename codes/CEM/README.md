# Consistency Enforcing Module (CEM)

An architectural model (implemented in PyTorch) that can wrap any existing, pre-trained, super-resolution model, for making its outputs consistent: Outputs of this module are guaranteed to match the low-resolution inputs, when downsampled using the given (or otherwise the bicubic) downsampling kernel.
<p align="center">
   <img src="fig_CEM_arch.png">
</p>

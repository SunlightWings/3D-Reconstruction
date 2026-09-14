# QuantSplat: Reliability-Aware Low-Bit VGGT for Sparse-View 3D Gaussian Splatting

QuantSplat studies how low-bit quantization of **VGGT** affects downstream
**3D Gaussian Splatting (3DGS)**, and whether lightweight learned modules can
detect or recover quantization-induced camera errors.

The project compares:

- **Full VGGT** — full-precision reference
- **W4A4 VGGT** — 4-bit weights / 4-bit activations
- **W3A3 VGGT** — aggressive 3-bit weights / 3-bit activations
- **W3A3 + residual camera correction**
- **W3A3 + learned confidence**
- **W3A3 + correction + confidence**

The central finding is that W3A3 camera degradation is **structured rather than
random**: focal-length error is especially large, can be predicted reliably,
and can often be corrected. However, better camera estimates do **not**
guarantee better final 3DGS rendering; downstream success also depends strongly
on scene geometry and view redundancy.

---

## Project pipeline

```text
Unposed RGB images
       |
       v
Full VGGT / W4A4 / W3A3
       |
       v
Camera intrinsics + extrinsics
Depth + dense world points
       |
       +-----------------------------+
       |                             |
       v                             v
Learned reliability          Residual camera corrector
predictor                    rotation / center / focal
       |                             |
       |                             v
       |                     corrected geometry
       |                             |
       +-------------+---------------+
                     |
                     v
           VGGT -> COLMAP adapter
                     |
                     v
                   3DGS
                     |
                     v
          held-out view rendering
                     |
                     v
         PSNR / SSIM / LPIPS / MAE

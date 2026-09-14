# Learned downstream modules

Raw images
  |
  v
QuantVGGT W3A3
  |
  +-----------------------------+
  |                             |
  v                             v
Confidence predictor       Residual corrector
(error/risk estimation)    (camera modification)
  |                             |
  | predicted:                  | modifies:
  | - rotation error            | - rotation
  | - center error              | - camera center
  | - focal error               | - focal parameters
  | - failure probability       |
  | - confidence                |
  |                             |
  +-------------+---------------+
                |
                v
             3DGS
                |
                v
       held-out novel-view render

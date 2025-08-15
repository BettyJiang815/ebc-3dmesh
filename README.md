# Edge-Boundary Conflict-Based Quality Evaluation for Textured Photogrammetric 3D Meshes

## Abstract

This repository implements the method proposed in  

**"Edge-boundary conflict-based quality evaluation for textured photogrammetric 3D meshes"**, 

a no-reference quality assessment approach for detecting and quantifying **edge-boundary conflicts** in textured 3D meshes.

The method consists of three hierarchical indicators:

1. **GNECI** (*Gradient-weighted Normalized Edge Conflict Index*): Quantifies texture–geometry inconsistency at the edge level.
2. **FECI** (*Face-Level Edge Conflict Index*): Identifies low-quality triangles (Low-Quality Faces, LQFs) based on the “weakest link” principle.
3. **Mesh-Level Statistics Framework**: Provides global quality distribution, adaptive thresholding (IQR), and cross-model comparability.

It requires no reference model and can be applied to:

- Multi-resolution mesh quality control
- Quality analysis of public semantic segmentation datasets
- Evaluation of boundary optimization effects

---

## Method Overview

### 1. Edge-Level GNECI

- **Gradient computation**:

  - Convert RGB to grayscale (with human-vision-based spectral weighting)  
  - Gaussian smoothing to reduce noise  
  - Sobel operator to compute gradient magnitude

- **Sampling strategy**:

  - Project mesh edges to texture space  
  - Sample gradient values along edges using **Bresenham’s algorithm**  
  - Shrink endpoints inward by `edge_shrink_pixels` (default 3) to avoid boundary artifacts

- **NECI calculation**:

- $$
  \text{NECI} = \frac{G_{\text{max}} - G_{\text{mean}}}{G_{\text{max}} + G_{\text{mean}} + \epsilon}
  $$

- **GNECI calculation**:

- $$
  \mathrm{GNECI} = \mathrm{NECI} \times G_{\max}
  $$

### 2. Face-Level FECI

- Take the maximum GNECI among the three edges of a triangle:

- $$
  \mathrm{FECI} = \max(\mathrm{GNECI}_1, \mathrm{GNECI}_2, \mathrm{GNECI}_3)
  $$

### 3. Mesh-Level Statistics

- Compute mean, standard deviation, and quartiles for FECI distribution

- Apply **IQR-based thresholding** to detect LQFs:

- $$
  T_{\text{IQR}} = Q_{75} + 1.5 \times (Q_{75} - Q_{25})
  $$

- Calculate:

  - $$
    R_{\text{LQF}}: \text{proportion of LQF counts}
    $$

  - $$
    R_{\text{LQA}}: \text{proportion of LQF areas}
    $$

---

## Installation

```bash
pip install numpy opencv-python pandas plyfile
```

## Usage

Configure parameters in `1_calculate.py`:

```python
GLOBAL_CONFIG = {
    "input_root_dir": "path/to/ply_root",
    "textures_directory": "path/to/textures_root",
    "output_root_dir": "path/to/output",
    "gneci_method": "neci_gmax",   
    "feci_method": "max_gneci",
    "feci_color_min": 0.0,
    "feci_color_max": 100.0,
    "unified_threshold_method": "global_iqr",
    "iqr_multiplier": 1.5,
    "edge_shrink_pixels": 3,
    "recursive_search": True
}
```

### Metric variants & how to choose color ranges

This code exposes two knobs that let you try different formulations of the edge/face metrics:

- `gneci_method`: how to weight NECI at the **edge** level

  - `neci_gmax`:
    $$
    \text{GNECI} = \text{NECI} \times G_{\text{max}}
    $$

  - `neci_gmean`: 
    $$
    \text{GNECI} = \text{NECI} \times G_{\text{mean}}
    $$

- `feci_method`: how to aggregate edges at the **face** level

  - `max_gneci`:
    $$
    \text{FECI} = \max(\text{GNECI}_1, \text{GNECI}_2, \text{GNECI}_3)
    $$

  - `max_neci`:  
    $$
    \text{FECI} = \max(\text{NECI}_1, \text{NECI}_2, \text{NECI}_3)
    $$

  - `max_neci_gmean`: 
    $$
    \text{FECI} = \max(\text{NECI}_1 \times G_{\text{mean},1}, \text{NECI}_2 \times G_{\text{mean},2}, \text{NECI}_3 \times G_{\text{mean},3})
    $$

**What changes across variants?**

- **NECI** is normalized to 0-1 by construction.
- Multiplying by a gradient statistic (***Gmax*** or ***Gmean***) **rescales** the magnitude of edge/face scores depending on image contrast and your Sobel configuration.
- Therefore, **FECI’s numeric range depends on the chosen variant** and on the scene’s gradients. This is expected.

**Paper setting (recommended):**

- We used `gneci_method="neci_gmax"` and `feci_method="max_gneci"` in the paper
- With this setting, using a fixed visualization range like `feci_color_min=0.0`, `feci_color_max=100.0` works well in practice for many scenes.

**If you choose other variants:**

- `max_neci` stays in [0,1] → a color range such as **0–1** (or 0–100 after multiplying by 100 for readability) is appropriate.
- `max_neci_gmean` behaves similarly to `max_gneci`; a **0–100** range is often fine, but you should confirm with data-driven percentiles (see below).

### Practical guidance for color mapping (`feci_color_min/max`)

Because the **absolute scale** of FECI changes with the chosen variant and with scene contrast:

1. **Maintain cross-model comparability**
   - For meaningful side-by-side comparisons, keep `feci_color_min` and `feci_color_max` **identical across all models within the same variant**.
   - Do **not** directly compare raw color maps between different metric variants (e.g., `max_neci` vs. `max_gneci`) unless you normalize the values.
2. **Data-driven selection (recommended)**
   - Run the analysis once, examine the distribution of FECI values, and set:
     - `feci_color_min` = 10th percentile (Q10) of FECI
     - `feci_color_max` = 90th percentile (Q90) of FECI
   - This approach clips both low-end noise and extreme outliers, yielding a **stable and interpretable colormap**.
   - For horizontal comparisons across multiple models, compute **a global Q10/Q90** across all models in the set and use that fixed range.
3. **Rules of thumb**
   - For `feci_method="max_neci"`: expect values in [0,1] → color range can be Q10–Q90 within that interval (or scaled to 0–100 for display).
   - For `max_gneci` / `max_neci_gmean`: ranges are typically larger; Q10–Q90 often falls roughly within 0–100, but confirm with your actual data.

### Example: Custom colormap mapping

The visualization applies a linear color interpolation to represent FECI values within the chosen range：

```python
# t = (feci_clamped - feci_min) / (feci_max - feci_min) ∈ [0, 1], normalized FECI value after applying feci_color_min/max
r = int(46 + (211 - 46) * t)
g = int(125 - (125 - 47) * t)
b = int(50 - (50 - 47) * t)
```

- **t**: Normalized FECI in [0,1] after clamping and scaling.
- **RGB interpolation**:
  - **Low-end color** (t=0): dark olive tone (46, 125, 50)
  - **High-end color** (t=1): bright reddish tone (211, 47, 47)
- This mapping provides a perceptually distinct gradient that emphasizes high-conflict regions without saturating the entire mesh.You may adjust these endpoint colors as needed to better match your visualization preferences or to highlight specific value ranges.

## Outputs

```perl
output_root/
├─ models/
│  ├─ <model_id>/
│  │  ├─ csv_results/      # Edge-level & face-level metrics (GNECI, FECI)
│  │  └─ colored_ply/      # FECI-colored visualization models
└─ summary/
   └─ global_quality_summary.csv
```

+ `csv_results`: Edge-level and face-level quality metrics for further analysis

+ `colored_ply`: PLY models colored by FECI values mapped to a fixed color range

+ `summary`: Global statistics and thresholding results

## Reproducibility Recommendations

+ **Color mapping range**: Keep `feci_color_min` / `feci_color_max` identical across experiments for comparability

+ **IQR parameter**: Recommended `iqr_multiplier = 1.5`

+ **Edge shrink**: `edge_shrink_pixels = 3` is suggested by sensitivity analysis in the paper

## Citation

If you use it in a scientific work, we kindly ask you to cite it:

```bibtex
@article{edgeboundaryconflict2025,
  title={Edge-boundary conflict-based quality evaluation for textured photogrammetric 3D meshes},
  author={Jiang, Yuying and Hu, Zhongwen and Zhang, Yinghui and Wang, Jingzhe and Zeng, Xiaoru and Wu, Guofeng},
  year={2025}
}
```


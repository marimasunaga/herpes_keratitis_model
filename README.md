# code_for_paper_R1

Code repository for image analysis of herpes keratitis lesions. This repository contains Python scripts for running the mathematical models, generating simulation datasets, and performing image processing, analyzing, and visualizing dendritic and geographic patterns lesion patterns.

## Overview

This repository contains all simulation and analysis codes used to generate the
figures and supplementary movies in the manuscript. Each directory corresponds
to a specific figure or movie and includes only
the scripts, data, and processing steps required for that particular result.

Across the repository, the following types of analyses appear (depending on the figure):
- Numerical simulations of lesion growth (Type A–E and modified model)
- Generation of raw grid images and time-evolution movies
- Alpha-shape–based contour extraction
- Skeletonization and branch detection
- Distance-map–based quantitative analysis
- Statistical comparisons of Root–Tip morphology
- CLIP-based feature extraction and UMAP visualization

This figure-by-figure organization allows users to directly trace each result
in the manuscript to the exact computation and code that produced it.

## Repository Structure

Directories are organized by figure and supplementary movie numbers in the manuscript.
Each folder contains only the scripts and data required to reproduce that specific result.

### Main Figures
- **Fig7/** – Scripts and results for Figure 7
- **Fig8/** – Scripts and results for Figure 8
- **Fig9/** – Scripts and results for Figure 9
- **Fig10/** – Scripts and results for Figure 10
- **Fig11/** – Full analysis pipeline, including:
  - Simulation output (`0round_cell_images/`)
  - Alpha-shape contour extraction (`1alpha_shape/`)
  - Dendritic/geographic classification (`2dendritic_images/`, `2geographic_images/`)
  - Distance map computation (`3distance_map_csv/`)
  - CSV post-processing (`4process_csv/`)
  - Plot generation (`5graph_distance_map/`, `5overlayed_distance_map/`)
  - Statistical analysis (`6analysis.py`, `6stat_graph/`)
- **Fig12/** – CLIP-based feature extraction and UMAP visualization
- **FigS4/** – Branch filtering workflow and exclusion count (Supplementary Fig. S4)
- **FigS5S6/** – Clinical image analysis (details in `FigS5S6/README.md`)

### Supplementary Movies
- **MovieS1–S5/** – Time-evolution movies for Type A–E
- **MovieS6/** – VZV model time-evolution
- **MovieS7/** – Multi-seed cytokine-gradient movie
- **MovieS8/** – Uniform-noise (heterogeneity-only) model

### Additional Data

- **`dt05dv005/`**: Supplementary simulations performed with a smaller time step (dt = 0.5).
  These results are included to confirm that the qualitative behavior of the model
  remains consistent with the main analysis (dt = 1).

## Requirements

### Core Dependencies

```bash
pip install opencv-python numpy pandas scikit-image scipy matplotlib seaborn shapely
```

### For CLIP Feature Extraction (Fig12)

```bash
pip install torch torchvision clip-by-openai pillow tqdm
```

### For Statistical Analysis

```bash
pip install statsmodels
```

### For UMAP Visualization (Fig12)

```bash
pip install umap-learn
```

## Usage

### Example: Processing Fig11 Data

```bash
cd Fig11

# Step 1: Generate alpha shapes
python 1alpha_shape.py

# Step 2: Classify dendritic/geographic patterns
python 2select_dendritic_pattern.py

# Step 3: Extract branches and generate distance maps
python 3distance_map_csv.py

# Step 4: Process CSV data
python 4process_csv.py

# Step 5: Generate graphs and overlays
python 5graph_distance_map.py
python 5overlayed_distance_map_images.py

# Step 6: Statistical analysis
python 6analysis.py
```

### Clinical Image Analysis (FigS5S6)

See `FigS5S6/README.md` for detailed instructions on clinical image processing.


## Output Files

- **CSV files**: Branch distance profiles (`*_branches.csv`, `*_branches_processed.csv`)
- **PNG images**: Processed images, graphs, overlays
- **NPZ files**: CLIP feature vectors (Fig12)
- **PDF/PNG**: Statistical plots and visualizations

## Notes

- Scripts are designed to process images in specific directory structures
- Most scripts automatically create output directories if they don't exist
- File paths in scripts may need to be adjusted based on your system
- Some scripts include interactive modes (e.g., clinical image analysis)

## Citation

If you use this code in your research, please cite the associated paper.


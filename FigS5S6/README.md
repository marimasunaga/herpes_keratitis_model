# Clinical Image Distance Map Analysis Scripts

This repository contains scripts for processing clinical images (cropped images) to perform skeletonization, extract distance map values, calculate root/tip values, and generate overlay images and graphs.

## Required Packages

Install the required Python packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or using a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

If using Jupyter notebook, run:

```python
!pip install -r requirements.txt
```

To install individually:

```bash
pip install opencv-python numpy pandas scikit-image scipy matplotlib
```

## File Structure

1. **clinical_distance_map_csv.py** - Skeletonization and distance map CSV generation
2. **clinical_graph_distance_map.py** - Graph generation (root/tip value extraction and visualization)
3. **clinical_overlayed_distance_map.py** - Overlay image generation

## Usage

### Step 1: Generate Distance Map CSV

```bash
python3 clinical_distance_map_csv.py
```

**Note**: Use `python3` if `python` command is not found.

**Interactive Mode (Default):**
- Each image is displayed with the skeleton overlaid (skeleton in green)
- Click to select Start position (blue marker)
- Click again to select Goal position (red marker)
- Press 'Enter' to confirm, 'r' to reset, 'q' to skip, 'Esc' to cancel
- Selected Start/Goal positions are saved in the CSV file's `path_id` column

This script:
- Processes all PNG images in the same directory
- Performs skeletonization
- Allows interactive selection of Start/Goal positions
- Calculates distance map values along the path between Start and Goal
- Outputs `clinical_branches.csv` with distance values for each path

### Step 2: Generate Graphs

```bash
python3 clinical_graph_distance_map.py
```

This script:
- Reads `clinical_branches.csv`
- Generates graphs for each path showing distance map values
- Calculates Root value (first local minimum) and Tip value (last 10-20%)
- Saves graphs to `graphs/` directory
- Saves Root/Tip values to `clinical_root_tip_values.csv`

### Step 3: Generate Overlay Images

```bash
python3 clinical_overlayed_distance_map.py
```

This script:
- Reads Start/Goal positions from `clinical_branches.csv`
- Overlays skeleton on the original image
- Color-codes Root region (light blue), Tip region (orange), and Middle region (gray)
- Marks Start position (blue) and Goal position (red)
- Saves overlay images to `overlayed_images/` directory

## Output Files

- `clinical_branches.csv` - Distance map values for each path (path_id, node_0, node_1, ...)
- `clinical_root_tip_values.csv` - Root and Tip values for each path
- `graphs/` - Graph images for each path
- `overlayed_images/` - Overlay images with color-coded regions

## Processing Flow

1. **Skeletonization**: Convert binary image to skeleton
2. **Interactive Selection**: Manually select Start and Goal positions on the skeleton
3. **Path Finding**: Find the shortest path between Start and Goal using BFS (breadth-first search)
4. **Distance Map Calculation**: Calculate distance transform values along the path
5. **Root/Tip Detection**:
   - Root: First local minimum in the smoothed distance curve
   - Tip: Last 10-20% of the path
6. **Visualization**: Generate graphs and overlay images

## Notes

- All scripts process images in the same directory where they are located
- Start/Goal positions are stored in the CSV file's `path_id` column in the format: `filename_startX_Y_goalX_Y`
- The scripts automatically create output directories (`graphs/`, `overlayed_images/`) if they don't exist

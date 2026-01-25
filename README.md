# DefectMaster Pro

A simple and user-friendly **GUI-based multi-label defect classification training tool**, designed specifically for few-shot and multi-class defect detection scenarios.  
With a graphical interface, you can complete the entire workflow in one click: Excel label conversion → optional bounding box annotation → data augmentation → model training → evaluation.

Key features:
- Multi-label classification with **EfficientNet-B0 + FPN + CBAM** attention mechanism
- Advanced data augmentation including **Topology-Preserving Elastic Deformation Augmentation (TPED)** and binary channel
- Per-class augmentation multiplier for handling imbalanced datasets
- Few-shot Ensemble Learning (multiple random seeds + averaging ensemble)
- Automatic visualization: Grad-CAM heatmaps, training curves, confusion matrices & error analysis
- Focal Loss, Early Stopping and learning rate scheduling

## Quick Start (Recommended Workflow)

### 1. Clone the Repository

```bash
git clone https://github.com/zokaraa/DefectMaster-Pro.git
cd DefectMaster-Pro
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Recommended environment:
- Python 3.8 – 3.11
- GPU recommended for faster training
- Works on Windows / macOS / Linux (tkinter is built-in on Windows; Linux/macOS may need to install tk separately)

### 3. Prepare Your Data (Most Important Step)
1. Put all training images (png / jpg / jpeg / bmp / tif, etc.) into one single folder, e.g.: `data/images/`
2. **Strongly recommended**: Use the provided template file [`sample dataset/defect statistics.xlsx`](sample%20dataset/defect%20statistics.xlsx)  
   (Modify this file with your own labels. Do **not** change column order or format.)
   
   - First column: `image id` (must be consecutive integers starting from 1, matching image filenames like `1.png`, `2.png`, …)
   - Following columns: defect class names (must **exactly match** the class names you will enter in the GUI)
   - Values: fill `1` if the class exists; leave blank if absent
     
   Example (4 classes):

   | image id | No Defect | Dislocation | Bridge | Junction |
   |----------|-----------|-------------|--------|----------|
   | 1        |           | 1           |        |          |
   | 2        | 1         |             |        |          |
   | 3        |           |             | 1      | 1        |

    **Important**: Do **not** change the column order or format arbitrarily, otherwise the program may fail to read correctly.

### 4. Launch the Program
```Bash
python "DefectMaster Pro.py"
```

### 5. GUI Operation Steps
1. **Image Folder**: Select the folder containing all images 
2. **Excel Label File**: Select your modified `.xlsx` file
3. **Class Names**: Enter class names (comma-separated, must match Excel column names exactly)
   Example: `No Defect,Dislocation,Bridge,Junction`
4. **Experiment Save Path**: Recommended to use the auto-generated path, or choose an empty folder manually
5. Click **Step 1: Convert Excel → Labels** (required)
6. (Optional) Click **Step 2: BBox Annotation** to draw bounding boxes
7. Adjust hyperparameters as needed (augmentation, lr, epochs, seeds…)
8. Click **Step 3: Start Training and Testing** to begin the full training pipeline (train + val + test)

Training progress and logs are shown in real-time at the bottom of the window. All results are automatically saved in the experiment directory after completion.

## Main Output Files (in the experiment directory)
```text
experiments/experiment_YYYYMMDD_HHMMSS/
├── augmentation_samples/      # Before/after augmentation sample images (debug & visualization)
├── confusion_matrices/        # Per-seed confusion matrix images (.png)
├── data_splits/               # Train/val/test split info for each seed (text/csv)
├── ensemble_error_samples/    # Error samples from ensemble model
├── heatmaps_alone/            # Standalone Grad-CAM heatmaps (.png)
├── metric_curves/             # Training/validation metric curves (.png)
├── metric/                    # Per-seed test metrics summary (csv)
├── model_error_samples/       # Per-seed model error sample images
├── models/                    # Best model weights per seed (.pth files)
├── summary/                   # Overall summary plots (F1 bar chart, etc.)
├── test_error_analysis/       # Test set error statistics (csv)
├── test_results/              # Test confusion matrices & metrics (csv/json)
├── training_curves/           # Loss & accuracy curves per seed (.png)
├── val_error_analysis/        # Validation set error statistics (csv)
├── data_split.txt             # General data split log
├── gui_log.txt                # GUI operation log
├── model.pth                  # Best single model weight (often the top seed)
└── training_log.json          # Full training history & summary (json)
```

## Dependencies (reference for requirements.txt)
```text
texttorch>=2.0.0
torchvision>=0.15.0
numpy>=1.23
pandas>=1.5
Pillow>=9.0
opencv-python>=4.7
tqdm>=4.64
scikit-learn>=1.2
seaborn>=0.12
matplotlib>=3.6
scipy>=1.9
iterative-stratification>=0.1
psutil>=5.9
```
**Note**: tkinter is required for GUI (usually included in Python; on Linux, install via `sudo apt install python3-tk`).

## Known Limitations & Notes
- Currently assumes image filenames are 1.png, 2.png, … (consecutive integers). Arbitrary filenames are not yet supported (can be improved later).
- Excel should have headers in the first row; the program automatically skips any statistic rows.
- Recommended class count: 2–10. Too many classes may lead to poor convergence.
- Memory usage depends on batch size, image resolution, and augmentation strength.
- If training hangs or crashes, check: All images exist in the folder; Excel format is correct; GPU memory is sufficient

## License
This project is licensed under the MIT License.

## Acknowledgements
- torchvision pretrained models
- iterative-stratification for multi-label splitting
- tkinter for lightweight GUI

Feel free to open issues, star, or contribute!

Last updated: January 2026

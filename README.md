# Installation and Usage Guide for SAM2-based River Channel Extraction

## Installation

1. Use the following command to install SAM2 on a GPU computer (python>=3.10, torch>=2.5.1, torchvision>=0.20.1):
```bash
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .

2.Download this code locally.

3.Place this code in the root directory of **SAM2**.

4.Ensure the following dependencies are installed:

- `numpy`
- `matplotlib`
- `pillow`
- `opencv-python`
- `rasterio`
- `scikit-image`
- `tk` *(optional, required if running on a Linux server)*

You can install these dependencies using:

```bash
pip install numpy matplotlib pillow opencv-python rasterio scikit-image
# Optional for Linux
# pip install tk

## Meandering River Channel Extraction
This code extracts river channels using SAM2 and converts them into binary images for subsequent geometric parameter extraction of narrow-banded meandering rivers in ArcGIS.

## Usage
1.Prepare remote sensing imagery in .tif format (with geospatial information) and the corresponding .jpg format.
2.Modify the following content in the script to meet your needs:
- `tiff_path: Geo-referenced .tif file`
- `input_image_path: Preprocessed .jpg image`
- `sam2_checkpoint: Path to the SAM2 model`
- `model_cfg: Configuration file corresponding to the model`
- `binary_mask_tiff_dir: Output directory for the results`
3.Run the script.
- `Left-click to mark the area of interest (foreground points).`
- `Right-click to mark the area not of interest (background points).`
- `To ensure segmentation accuracy, you can click as many foreground and background points as needed.`
- `model_cfg: Configuration file corresponding to the model`
- `After clicking, close the visualization window`
4.The .jpg image will be segmented, and the geospatial information from the .tif image will be assigned to the segmentation result.
5.The prediction result will be saved as a .tif image.

# intelligent-extraction-of-river
## River Channel Extraction

By importing raster TIFF image data and employing an interactive foreground and background sample point annotation approach, the Segment Anything Model (SAM) is invoked to perform intelligent segmentation of river channel features, generating high-precision binary masks for accurate extraction of fluvial elements.

## Installation

Built upon SAM2 - installation complies with SAM2's official requirements:  
https://github.com/facebookresearch/segment-anything

### Prerequisites

Make sure you have installed the following dependencies:

#### Core Packages
- NumPy
- OpenCV (cv2)
- Matplotlib
- Rasterio
- Pillow (PIL)

#### SAM2-Specific
- segment-anything==2.0
- SAM2 model weights

#### System Libraries
- GDAL (for geospatial processing)

### Installation

You can install the Python dependencies using this command:

```bash
pip install torch numpy opencv-python matplotlib rasterio pillow


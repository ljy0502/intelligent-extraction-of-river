Installation
1. Use the following command to install SAM2 on a GPU machine (python>=3.10 torch>=2.5.1 torchvision>=0.20.1):
  git clone https://github.com/facebookresearch/sam2.git && cd sam2
  pip install -e .

2. Download this code to your local machine

3. Place this code in the root directory of SAM2

4. Make sure the following dependencies are installed:
•	numpy
•	matplotlib
•	pillow
•	opencv-python
•	rasterio
•	scikit-image
•	tk (optional, required if running on a Linux server)

You can install these dependencies using the following command:
pip install numpy matplotlib pillow opencv-python rasterio scikit-image


Meandering River Channel Extraction
This code is used to extract river channels from remote sensing images using SAM2 and convert them into binary images. The subsequent extraction of geometric parameters of narrow meandering rivers in ArcGIS will be based on this segmentation result.

Usage
1. Prepare remote sensing images in both .tif format (with geospatial information) and corresponding .jpg format

2. Modify the following script parameters to suit your needs:
•	tiff_path: Geo-referenced .tif file
•	input_image_path: Preprocessed .jpg image path
•	sam2_checkpoint: SAM2 model checkpoint
•	model_cfg: Configuration file corresponding to the model
•	binary_mask_tiff_dir: Output directory for results

3. Run the script, left-click to mark regions of interest (foreground points), right-click to mark regions not of interest (background points). To ensure segmentation accuracy, you can click as many foreground and background points as possible. After finishing point selection, close the visualization window.

4. The .jpg image will be segmented, and the geospatial information of the .tif image will be assigned to the segmentation result.

5. The prediction result will be saved as a .tif image

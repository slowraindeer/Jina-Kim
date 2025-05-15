# StyleAlignNet: Standardizing Annotator Style for Improved Segmentation Performance
This repository presents **StyleAlignNet**, a model designed to align various annotation styles in image segmentation tasks. StyleAlignNet standardizes segmentation masks with diverse styles to match a given reference style using a simple architectural unit and five key loss functions.


## Overview
StyleAlignNet operates by transforming all segmentation masks to conform to a user-specified reference style. Each annotator typically segments a region of interest (ROI) in their own distinct style, which can introduce inconsistency across the dataset. To address this, StyleAlignNet uses a subset of the dataset annotated by multiple annotators to learn the unique characteristics of each annotation style and the differences between them. Through this training process, the model learns how to convert segmentation masks from one annotator’s style to another, enabling consistent and standardized annotations across the dataset.


## File Structure

- `StyleAlignNet.py` : Contains the implementation of the StyleAlignNet model. You can modify the directory structure as needed. The model expects input images of size 512×512×3, so appropriate image preprocessing is required.
- `Custom_loss.py` : Defines the loss functions used in `StyleAlignNet.py`.
- `README.md` : Provides an overview of StyleAlignNet and usage instructions.

##Installation
Clone the repository:

```bash
git clone https://github.com/slowraindeer/StyleAlignNet
cd StyleAlignNet

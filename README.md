# DeepCSeg
Source code for DeepCSeg plug-in for CellProfiler

Author: Wenchao Han, University of Toronot, Sunnybrook Research Institute, Toronto. Contact: wenchao.han@sri.utoronto.ca

The plug-in package for implementation can be found: https://drive.google.com/drive/folders/1WBYFH9bf89s-xjQNZHKSGdFov08h0iFG?usp=sharing.
The details for the methodology and usability of DeepCSeg can be found in paper: 'DeepCSeg: Whole cell segmentation of immunofluorescence multiplexed images using Mask R-CNN'
One example video for implementation can be found: https://drive.google.com/file/d/1mxQ3yOWj45AAwUwi439mOz207A5SEtjD/view?usp=sharing.
Plug-in installation and setup can be found: https://youtu.be/sirRJc-A4tc.

This repository includes two code files: 1) DeepCSeg.py 2) DeepCSeg plug-in.py

The DeepCSeg.py is the source code for generating the DeepCSeg executable. This code includes functions from MaskRCNN implementation developed by Matterport Inc. under the MIT license. To generate the executable for implementation, the developer may execute the code 'pyinstaller -F DeepCSeg.py'. The pyinstaller needs to be installed in the environment. Also, the environment needs to meet the requirements for the MaskRCNN implementation. Please refer https://github.com/matterport/Mask_RCNN for configuration details.

The DeepCSeg plug-in.py is the python plug-in code for use in CellProfiler.


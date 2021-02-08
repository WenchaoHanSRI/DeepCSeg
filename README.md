# DeepCSeg
Source code for DeepCSeg plug-in for CellProfiler

Author: Wenchao Han, Medical Biophysics, University of Toronto, Biomarker Imaging Research Laboratory, Sunnybrook Research Institute, Toronto. email: wenchao.han@sri.utoronto.ca

The plug-in package for implementation can be found: https://drive.google.com/drive/folders/1WBYFH9bf89s-xjQNZHKSGdFov08h0iFG?usp=sharing.
Details for methodology and usability of DeepCSeg can be found in paper: 'DeepCSeg: Whole cell segmentation of immunofluorescence multiplexed images using Mask R-CNN'.
The video for implementation on an example case can be found: https://drive.google.com/file/d/1mxQ3yOWj45AAwUwi439mOz207A5SEtjD/view?usp=sharing.
The video for Plug-in installation and setup can be found: https://youtu.be/sirRJc-A4tc.

This repository includes two code files: 1) DeepCSeg.py 2) DeepCSeg plug-in.py

The DeepCSeg.py is the source code for generating the DeepCSeg executable. To generate the executable for implementation, users may execute the code 'pyinstaller -F DeepCSeg.py'. The 'pyinstaller' and 'flask' need to be installed in the environment. This source code includes functions from MaskRCNN implementation developed by Matterport Inc. under the MIT license. The environment needs to meet the requirements for the MaskRCNN implementation. Please refer https://github.com/matterport/Mask_RCNN for configuration details.

For Mac/Linux users, the executable file needs to be generated from the source code in Mac/Linux with all the dependencies/libraries mentioned above. 

The DeepCSeg plug-in.py is the python plug-in code for use in CellProfiler.


# DeepCSeg
Source code for DeepCSeg plug-in for CellProfiler

Author: Wenchao Han, Biomarker Imaging Research Laboratory(BIRL) https://sunnybrook.ca/research/content/?content=sri-cerigt-birl-about, Sunnybrook Research Institute, Medical Biophysics, University of Toronto, Toronto, ON, Canada.
email: wenchao.han@sri.utoronto.ca

The plug-in package for implementation can be found: https://drive.google.com/drive/folders/1WBYFH9bf89s-xjQNZHKSGdFov08h0iFG?usp=sharing.
Details for methodology and usability of DeepCSeg can be found in paper: 'DeepCSeg: Whole cell segmentation of immunofluorescence multiplexed images using Mask R-CNN'.
The video for implementation on an example case can be found: https://youtu.be/gpLDjPQJF8Q.
The video for Plug-in installation and setup can be found: https://youtu.be/sirRJc-A4tc.

.center[

![My image](https://user-images.githubusercontent.com/60233311/115458278-23816080-a1eb-11eb-8b3b-7879bf9a474d.png)

.caption[
**Ovarian cancer tissue example** Image caption
]

]


This repository includes two code files: 1) DeepCSeg.py 2) DeepCSeg plug-in.py

The DeepCSeg.py is the source code for generating the DeepCSeg executable. To generate the executable for implementation, users may execute the code 'pyinstaller -F DeepCSeg.py'. The 'pyinstaller' and 'flask' need to be installed in the environment. This source code includes functions from MaskRCNN implementation developed by Matterport Inc. under the MIT license. The environment needs to meet the requirements for the MaskRCNN implementation. Please refer https://github.com/matterport/Mask_RCNN for configuration details.

For Mac/Linux users, the executable file needs to be generated from the source code in Mac/Linux with all the dependencies/libraries mentioned above. 

The DeepCSeg plug-in.py is the python plug-in code for use in CellProfiler.

# Citation
Use this bibtex to cite this repository:
```
@misc{DeepCSeg_2021,
  title={DeepCSeg, a CellProfiler plug-in for whole cell segmentation for immunofluorescence multiplexed images},
  author={Wenchao Han},
  year={2021},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/WenchaoHanSRI/DeepCSeg}},
}
```


"""
The plug-in performs instance cell and nucleus segmentation using MaskRCNN,
and uses the segmentation results to compute cellular statistics for a given
bio-marker stained image.
"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.transform
import skimage.color
import skimage.filters
import os
import pickle
import skimage.io
import urllib.request
import json
import pickle
import csv
import random
import colorsys
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import sys

# define functions for unpack and display segmentation results
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
       This function is reused from the implementation created by Matterport Inc.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    This function is reused from the implementation created by Matterport Inc.

    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    # fig.tight_layout()
    return ax

def display_instances_single(image, boxes, masks, class_ids, class_names, color,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      captions=None):
    """
    This function overlays the segmentation results on the image in a form of contours.
    This function is reused from the implementation created by Matterport Inc.
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    print('#')
    print('#')

    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True


    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = color
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        #Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    return ax, masked_image

def unpack_mask_indecies(r, padding=None):
    """
    This function unpack the segmentation results.

    Args:
        r[dictionary]: segmentation results including all information
        padding[list]: list of padding sizes

    Returns:
        recon_mask[ndarray, dtype: bool]: a stack of masks of segmented regions
        number_instance[int]: the number of the masks
    """
    masks_size = r['maskssize']
    mask_ori_size = r['masksorisize']
    recon_mask = np.zeros((mask_ori_size[0], mask_ori_size[1], masks_size[2]), dtype=bool)
    masksIdx = r['masksindexori']
    for j in range(0, len(masksIdx)):
        idx = masksIdx[j]
        if padding:
            recon_mask[idx[0]-padding[0][1], idx[1]-padding[1][1], j] = True
        else:
            recon_mask[idx[0], idx[1], j] = True
    number_instance = len(masksIdx)
    return recon_mask, number_instance

def find_matching_nucleus(mask_int, recon_mask2, number_instance2):
    """
    This function finds the nucleus that matches the reference cell by finding the nucleus that have the largest
    overlap of the reference cell.

    Args:
        mask_int[ndarray: dytpe: int]: reference cell mask
        recon_mask2[ndarray, dtype: bool]: a stack of cell masks
        number_instance2[int] : numbers of nucleus masks to loop through

    Returns:
        mask2_nucleus_int[ndarray, dtype: int]: the best matching nucleus mask
    """
    sum_G = []
    for j in range(0, number_instance2):
        mask2 = recon_mask2[:, :, j]
        mask2_int = mask2.astype(int)
        mask_diff = mask_int * mask2_int
        sum_masked = mask_diff.sum()
        sum_G.append(sum_masked)
    id_max = sum_G.index(max(sum_G))
    if sum_G[id_max] > 0:
        mask2_nucleus = recon_mask2[:, :, id_max]
        mask2_nucleus_int = mask2_nucleus.astype(int)
    else:
        mask2_nucleus_int = np.zeros((recon_mask2.shape[0], recon_mask2.shape[1]), dtype=int)
    return mask2_nucleus_int

def compute_maskedregion_metric(mask_image, mask_int):
    """
    This function computes some of the the cellular statistics.

    Args:
        mask_image[ndarray, dtype:int 32]: testing image masked by the mask_int
        mask_int[ndarray, dtype:int 32]: a region mask

    Returns:
        computed cellular metrics for an instance of masked region
    """
    MaxIntensity = np.amax(mask_image)
    MinIntensity = np.amin(mask_image)
    Total = np.sum(mask_image)
    Region = mask_image[np.nonzero(mask_image)]
    Median = np.median(Region)
    Std = np.std(Region)
    regionMetrics = skimage.measure.regionprops(mask_int)
    try:
        area = regionMetrics[0].area
        Mean = Total / area
    except:
        area = 0
        Mean = float('nan')
    return MaxIntensity, Total, Median, Std, Mean, area

# define default names for output files
NucleiResultName = ["NucleiName"]
CellResultName = ["CellName"]
MetricFileName = ["BiomarkerMetric"]

# define the plug-in module
class DeepCSeg(cellprofiler.module.ImageProcessing):
    # category = "Image Segmentation"

    module_name = "DeepCSeg"

    variable_revision_number = 1

    # create settings for the cellprofiler plug-in
    def create_settings(self):

        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            u"DAPI",
            doc="""
                A DAPI channel image.
                """
        )

        self.y_name = cellprofiler.setting.ImageNameSubscriber(
            u"MEM",
            doc="""
                A membrane channel image.
                """
        )

        self.z_name = cellprofiler.setting.ImageNameSubscriber(
            u"Marker",
            doc="""
                An image stained with the biomarker for quantitative analysis.
                """
        )

        self.modle_drive_letter = cellprofiler.setting.DirectoryPath(
            "Model path",
            doc=""""
            The path where the trained model saved. There are two trained models for this module. One is a trained model for
            nuclei segmentation, and the other one is a trained model for cell segmentation.
            """
        )

        self.display_choice = cellprofiler.setting.Choice(
            "Display segmentation on bio-marker image",
            [
                'YES',
                'NO'
            ],
            doc="""User choose to run the cell segmentation or not. The module will not re-run the segmentation if the segmentation results already exited
            in the result folder from previous analysis, and the user choose to run the cell segmentation. Otherwise, the module executes the cell segmentation."""
        )

        self.segmentation_choice = cellprofiler.setting.Choice(
            "Running cell segmentation",
            [
                'YES',
                'NO'
            ],
            doc="""User choose to run the cell segmentation or not. The module will not re-run the segmentation if the segmentation results already exited
            in the result folder from previous analysis, and the user choose to run the cell segmentation. Otherwise, the module executes the cell segmentation."""
        )

        self.pathname = cellprofiler.setting.DirectoryPath(
            "Output file location"
        )

        self.Nuclei_result = cellprofiler.setting.FileImageNameProvider(
            "Nuclei file name",
            NucleiResultName[0],
            doc="""NucleiResultName.
                """
        )

        self.Cell_result = cellprofiler.setting.FileImageNameProvider(
            "Cell file name",
            CellResultName[0],
            doc="""CellResultName.
                """
        )

        self.Biomarker_filename = cellprofiler.setting.FileImageNameProvider(
            "Biomarker metric filename",
            MetricFileName[0],
            doc="""the filename which provides the metrics calculated from the biomarker image.
                """
        )

    # set the settings
    def settings(self):

        result = [
            self.x_name,
            self.y_name,
            self.z_name,
            self.modle_drive_letter,
            self.display_choice,
            self.segmentation_choice,
            self.pathname,
            self.Nuclei_result,
            self.Cell_result,
            self.Biomarker_filename
        ]
        return result

    # define settings that are visible in the plug-in
    def visible_settings(self):

        result = [
            self.x_name,
            self.y_name,
            self.z_name,
            self.modle_drive_letter,
            self.display_choice,
            self.segmentation_choice,
            self.pathname,
            self.Nuclei_result,
            self.Cell_result,
            self.Biomarker_filename
        ]

        return result

    # executing the code
    def run(self, workspace):

        x_name = self.x_name.value  # DAPI image name
        y_name = self.y_name.value  # MEM image name
        z_name = self.z_name.value  # Biomarker image name

        # this is an image object defined in CellProfiler
        images = workspace.image_set

        # encoding input files information for exporting to the executable
        x = images.get_image(x_name)  # DAPI image object
        y = images.get_image(y_name)  # MEM image object
        z = images.get_image(z_name)  # Biomarker image object

        # extracting image paths
        path_dataset_DAPI = x.path_name
        path_dataset_MEM = y.path_name
        path_dataset_MARKER = z.path_name

        # Making output file names for nucleus and cell segmentation results
        NUCLEIFILENAME = self.Nuclei_result.value + '.dat'
        CELLFILENMAE = self.Cell_result.value + '.dat'

        # extracting input image names
        VAL_IMAGE_filename = x.file_name
        VAL_MEM_filename = y.file_name
        MARKER_filename = z.file_name

        # getting input information ready for the input files for the executable
        VAL_IMAGE_IDS = [VAL_IMAGE_filename.split('.', 1)[0]]  # DAPI image ID
        VAL_MEM_ID = [VAL_MEM_filename.split('.', 1)[0]]       # MEM image ID
        VAL_BIOMARKER_ID = [MARKER_filename.split('.', 1)[0]]  # BioMarker image ID
        SEGMENTATION = self.segmentation_choice.value          # if perform segmentation
        MODEL_PATH = self.modle_drive_letter.value        # directory where the trained model saved
        RESULT_PATH = self.pathname.value                 # result directory directory of the plug-in folder

        # put all the information to dataset [dictionary] for encoding
        dataset = {'path_dataset_DAPI': path_dataset_DAPI,
                   'path_dataset_MEM': path_dataset_MEM,
                   'path_dataset_Marker': path_dataset_MARKER,
                   'image_id': VAL_IMAGE_IDS,
                   'MEM_id': VAL_MEM_ID,
                   'MARKER_id': VAL_BIOMARKER_ID,
                   'segmentation': SEGMENTATION,
                   'nuclei_filename': NUCLEIFILENAME,
                   'cell_filename': CELLFILENMAE,
                   'model_drive_letter': MODEL_PATH,
                   'result_path': RESULT_PATH
                   }

        # encoding the dataset[dictionary] in json data format
        data = json.dumps(dataset)

        # encoding the strings using 'utf-8'- (unicode transformation format-8-bit)
        edata = data.encode('utf-8')

        headers = {}
        headers['Content-Type'] = 'application/json'

        # sending request with the input file information and getting response back after performing the segmentation
        # from the executable
        req = urllib.request.Request('http://localhost:5000/myplugin', edata, headers)

        # unpack the results from the response
        with urllib.request.urlopen(req) as response:
            the_page = json.load(response)

        # retrieving the file names of the output
        image_name = the_page['image']  # synthesized image file name (the synthesized image is a R-G-B image that is synthesized from DAPI and MEM images)
        infer_name = the_page['infer']  # cell segmentation result file name
        infer2_name = the_page['infer2']  # nucleus segmentation result file name
        image_bio_name = the_page['imagebio']  # the biomarker image file name[.dat]
        biomarker_path = the_page['biomarkerpath']  # the biomarker image full file path for computing cellular statistics
        RESULT_PATH = the_page['RESULT_PATH']  # result folder name

        # retrieving the the results generated from the executable (core algorithm)
        with open(infer_name, 'rb') as a:
            r = pickle.load(a)               # loading cell segmentation results
        with open(infer2_name, 'rb') as b:
            r2 = pickle.load(b)              # loading nucleus segmentation results
        with open(image_name, 'rb') as e:
            image = pickle.load(e)           # loading synthesized color image
        with open(image_bio_name, 'rb') as f:
            image_bio = pickle.load(f)       # loading biomarker image for displaying

        print('get results successfully')

        # unpack the mask indices. recon_mask is mask indices for segmented cells, recon_mask2 is for segmented nuclei
        recon_mask, number_instance = unpack_mask_indecies(r)     # unpack cell segmentation results
        recon_mask2, number_instance2 = unpack_mask_indecies(r2)  # unpack nucleus segmentation results

        # display the segmentation results for cell and nucleus segmentation by
        # overlaying segmentation results on the synthesized color images
        print('display results')
        color = (0.8, 0.5, 0.0)   # defining cell segmentation contour color (orange) for display
        # overlay cell segmentation contours on the color image
        ax1, images = display_instances_single(image, r['rois'], recon_mask, r['class_ids'],
                                               ['background', 'cell'], color, None, ax=get_ax(1),
                                               show_bbox=False, show_mask=False,
                                               title="Predictions")
        color = (0.0, 0.5, 1.0)   # defining nucleus segmentation contour color (blue) for display
        # overlay nucleus segmentation contours on the color image
        ax2, marked_image = display_instances_single(image, r2['rois'], recon_mask2, r2['class_ids'],
                                                     ['background', 'cell'], color, None, ax=ax1,
                                                     show_bbox=False, show_mask=False,
                                                     title="Predictions")
        ax2.imshow(images.astype(np.uint8))
        plt.show()
        print('display successfully')

        # display the segmentation results on the bio-marker image
        if self.display_choice.value == 'YES':
            print('display results')
            color = (0.8, 0.5, 0.0)
            ax1_test, images_test = display_instances_single(image_bio, r['rois'], recon_mask, r['class_ids'],
                                                   ['background', 'cell'], color, None, ax=get_ax(1),
                                                   show_bbox=False, show_mask=False,
                                                   title="Predictions")
            color = (0.0, 0.5, 1.0)
            ax2_test, marked_image_test = display_instances_single(image_bio, r2['rois'], recon_mask2, r2['class_ids'],
                                                         ['background', 'cell'], color, None, ax=ax1_test,
                                                         show_bbox=False, show_mask=False,
                                                         title="Predictions")
            ax2_test.imshow(images_test.astype(np.uint8))
            plt.show()
            print('display successfully')

        # compute the cellular statistics for the bio-marker image
        # defining the result file name for saving
        csvname = self.Biomarker_filename.value + ".csv"  # the csv file name for writting computed cellular statistics
        csvfilename = os.path.join(RESULT_PATH, csvname)  # full file for the csv file
        TestImage = skimage.io.imread(biomarker_path)     # read testing image using the full file path
        if len(TestImage.shape) > 2:   # if the channel is more than one, use the first channel
            TestImage = TestImage[:, :, 0]

        # creating csv file for cellular statistics output
        with open(csvfilename, 'w') as csvfile:
            fieldnames = ['cellID', 'perimeter', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Position_X',
                          'Position_Y', 'Cell_Area', 'Nucleus_Area', 'Cell_Max', 'Cell_Mean', 'Cell_Std', 'Cell_Median',
                          'Cell_TotalIntensities',  'Nucleus_Max', 'Nucleus_Mean', 'Nucleus_Std',
                          'Nucleus_Median', 'Nucleus_TotalIntensities']  # define the metrics to compute
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()
            # looping through each cell to compute cellular statistics and write the results to the CSV file
            for i in range(0, number_instance):
                mask = recon_mask[:, :, i]  # stack of segmentation mask indices
                mask_int = mask.astype(int)  # make sure the indices is integers
                # finding nucleus mask for the referring cell mask
                mask2_nucleus_int = find_matching_nucleus(mask_int, recon_mask2, number_instance2)
                mask_image = mask_int * TestImage   # masking the testing image using the cell mask
                mask_image_nucleus = mask2_nucleus_int * TestImage  # masking the testing image using the nucleus mask
                # compute cellular statistics
                CellMax, CellTotal, CellMedian, CellStd, CellMean, CellArea = compute_maskedregion_metric(mask_image, mask_int)
                # compute nucleus statistics
                NucleusMax, NucleusTotal, NucleusMedian, NucleusStd, NucleusMean, NucleusArea = compute_maskedregion_metric(mask_image_nucleus, mask2_nucleus_int)
                # compute the region based metrics using cellular mask
                regionMetrics = skimage.measure.regionprops(mask_int)
                centroid = regionMetrics[0].centroid
                perimeter = regionMetrics[0].perimeter
                eccentricity = regionMetrics[0].eccentricity
                major_axis_length = regionMetrics[0].major_axis_length
                minor_axis_length = regionMetrics[0].minor_axis_length
                # write all the computed results to the CSV file
                writer.writerow({'cellID': i, 'perimeter': perimeter,
                                 'Eccentricity': eccentricity,
                                 'MajorAxisLength': major_axis_length,
                                 'MinorAxisLength': minor_axis_length, 'Position_X': centroid[1],
                                 'Position_Y': centroid[0], 'Cell_Area': CellArea, 'Nucleus_Area': NucleusArea,
                                 'Cell_Max': CellMax, 'Cell_Mean': CellMean,
                                 'Cell_Std': CellStd, 'Cell_Median': CellMedian, 'Cell_TotalIntensities': CellTotal,
                                 'Nucleus_Max': NucleusMax, 'Nucleus_Mean': NucleusMean,
                                 'Nucleus_Std': NucleusStd, 'Nucleus_Median': NucleusMedian,
                                 'Nucleus_TotalIntensities': NucleusTotal
                                 })
                print('writing instantce', i)

    def display(self, workspace, figure):
        """Display the results, and possibly intermediate results, as
        appropriate for this module.  This method will be called after
        run() is finished if self.show_window is True.

        The run() method should store whatever data display() needs in
        workspace.display_data.  The module is given a CPFigure to use for
        display in the third argument.
        """
        figure.Close()  # modules that don't override display() shouldn't
        # display anything










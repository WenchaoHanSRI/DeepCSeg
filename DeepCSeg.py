from flask import request, Flask

# Flask constructor, which takes the name of the current module
app = Flask(__name__)

# define decorator. The decorator binds with the defined function 'home()'
@app.route("/myplugin", methods = ['GET', 'POST'])
def home():
    import json
    import pickle
    import os
    import sys
    import datetime
    import re
    import keras.backend as K
    import keras.layers as KL
    import keras.engine as KE
    import keras.models as KM
    import logging
    import random
    import numpy as np
    import tensorflow as tf
    import cv2
    from mrcnn.config import Config
    import math
    from collections import Counter

    ROOT_DIR = os.path.abspath("../../")  # get the root path
    sys.path.append(ROOT_DIR)  # add root directory to path

    # unpack the input information from request
    response = request.json      # get data info from request
    DATASET_DIR = response['path_dataset_DAPI']      # dapi image directory
    DATASET_DIR_MEM = response['path_dataset_MEM']   # membrane image directory
    DATASET_DIR_BIOMARKER = response['path_dataset_Marker']    # bio-marker image directory
    SEGMENTATION = response['segmentation']   # if performing segmentation or not, if not using the existing segmentation resutls
    RESULT_NUCLEI = response['nuclei_filename']   # result filename for nucleus segmentation
    RESULT_CELL = response['cell_filename']      # result filename for cell segmentation
    VAL_DAPI_IDS = response['image_id']          # dapi image name
    VAL_MEM_ID = response['MEM_id']              # membrane image name
    MODEL_PATH = response['model_drive_letter'].split('|')[1]  # directory where the trained models saved
    RESULT_PATH = response['result_path'].split('|')[1]   # results directory
    VAL_IMAGE_IDS = [VAL_DAPI_IDS[0]+'_COLOR']   # synthesized color image file name for performing cell segmentation
    VAL_BIOMARKER_ID = response['MARKER_id']  # test biomarker image name

    def FindImgFileName (DATASET_DIR, IMG_IDS):
        """
        This finds the image file name using the directory name and image name without the image format extension
        (e.g. .png,.jpg,.tif etc). This allows the plug-in reads images without limiting to one image format.

        Args:
            DATASET_DIR[str]: directory where holds the input image file
            IMG_IDS[list, dtypes: string]: input image name without format extension

        Returns:
            IMG_filename[string]: input image full filename
        """
        files = sorted(os.listdir(DATASET_DIR))
        matching = [s for s in files if IMG_IDS[0] in s]
        if len(matching) > 1:
            match_value_G = []
            for ID in matching:
                match_value = sum((Counter(ID) & Counter(IMG_IDS[0])).values())
                match_percent = match_value/len(ID)
                match_value_G.append(match_percent)
            max_value = max(match_value_G)
            max_index = match_value_G.index(max_value)
            IMG_filename = matching[max_index]
        elif len(matching) == 1:
            IMG_filename = matching[0]
        else:
            raise NameError('check input image name')
        return IMG_filename

    # read and check if reading the input images correctly
    DAPI_filename = FindImgFileName(DATASET_DIR, VAL_DAPI_IDS)  # get the dapi image filename
    MEM_filename = FindImgFileName(DATASET_DIR, VAL_MEM_ID)   # get the mem image filename
    BIOMARKER_filename = FindImgFileName(DATASET_DIR_BIOMARKER, VAL_BIOMARKER_ID)  # get the biomarker image filename
    DAPI_path = os.path.join(DATASET_DIR, DAPI_filename)  # dapi image full path
    MEM_path = os.path.join(DATASET_DIR_MEM, MEM_filename)  # membrane image full path
    BIOMARKER_path = os.path.join(DATASET_DIR_BIOMARKER, BIOMARKER_filename)  # biomarker image full path
    image_dapi = cv2.imread(DAPI_path)  # read dapi image
    if image_dapi is None:  # check if it reads successfully; cv2.imread returns none when it does not read image properly
        raise RuntimeError(DAPI_path, 'input image file cannot be read correctly')
    image_MEM = cv2.imread(MEM_path)  # read membrane image
    if image_MEM is None:
        raise RuntimeError(MEM_path, 'input image file cannot be read correctly')
    image_BIOMARKER = cv2.imread(BIOMARKER_path)  # read membrane image
    if image_BIOMARKER is None:
        raise RuntimeError(BIOMARKER_path, 'input image file cannot be read correctly')
    singlechannle_dapi = image_dapi[:, :, 0]  # take one channel of the input image (grey-level).
    singlechannel_MEM = image_MEM[:, :, 0]    # one channel of membrane image
    colorimage = np.dstack(
        (singlechannle_dapi, singlechannel_MEM, singlechannle_dapi))  # synthesize dapi and mem images to color image

    # compute the width and height of the input image for setting maximum instance numbers in config
    HightOfImg = colorimage.shape[0]
    WidthOfImg = colorimage.shape[1]

    ############################################################
    #  Dataset
    ############################################################

    class Dataset(object):
        """The base class for dataset classes.
        To use it, create a new class that adds functions specific to the dataset
        you want to use.
        """

        def __init__(self, class_map=None):
            self._image_ids = []
            self.image_info = []
            # Background is always the first class
            self.class_info = [{"source": "", "id": 0, "name": "BG"}]
            self.source_class_ids = {}

        def add_class(self, source, class_id, class_name):
            assert "." not in source, "Source name cannot contain a dot"
            # Does the class exist already?
            for info in self.class_info:
                if info['source'] == source and info["id"] == class_id:
                    # source.class_id combination already available, skip
                    return
            # Add the class
            self.class_info.append({
                "source": source,
                "id": class_id,
                "name": class_name,
            })

        def add_image(self, source, image_id, path, **kwargs):
            image_info = {
                "id": image_id,
                "source": source,
                "path": path,
            }
            image_info.update(kwargs)
            self.image_info.append(image_info)

        def image_reference(self, image_id):
            """Return a link to the image in its source Website or details about
            the image that help looking it up or debugging it.

            Override for your dataset, but pass to this function
            if you encounter images not in your dataset.
            """
            return ""

        def prepare(self, class_map=None):
            """Prepares the Dataset class for use.

            TODO: class map is not supported yet. When done, it should handle mapping
                  classes from different datasets to the same class ID.
            """

            def clean_name(name):
                """Returns a shorter version of object names for cleaner display."""
                return ",".join(name.split(",")[:1])

            # Build (or rebuild) everything else from the info dicts.
            self.num_classes = len(self.class_info)
            self.class_ids = np.arange(self.num_classes)
            self.class_names = [clean_name(c["name"]) for c in self.class_info]
            self.num_images = len(self.image_info)
            self._image_ids = np.arange(self.num_images)

            # Mapping from source class and image IDs to internal IDs
            self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                          for info, id in zip(self.class_info, self.class_ids)}
            self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                          for info, id in zip(self.image_info, self.image_ids)}

            # Map sources to class_ids they support
            self.sources = list(set([i['source'] for i in self.class_info]))
            self.source_class_ids = {}
            # Loop over datasets
            for source in self.sources:
                self.source_class_ids[source] = []
                # Find classes that belong to this dataset
                for i, info in enumerate(self.class_info):
                    # Include BG class in all datasets
                    if i == 0 or source == info['source']:
                        self.source_class_ids[source].append(i)

        def map_source_class_id(self, source_class_id):
            """Takes a source class ID and returns the int class ID assigned to it.
            """
            return self.class_from_source_map[source_class_id]

        def get_source_class_id(self, class_id, source):
            """Map an internal class ID to the corresponding class ID in the source dataset."""
            info = self.class_info[class_id]
            assert info['source'] == source
            return info['id']

        @property
        def image_ids(self):
            return self._image_ids

        def source_image_link(self, image_id):
            """Returns the path or URL to the image.
            Override this to return a URL to the image if it's available online for easy
            debugging.
            """
            return self.image_info[image_id]["path"]

        def load_image(self, image_id):
            """Load the specified image and return a [H,W,3] Numpy array.
            """
            # Load image
            print(self.image_info[image_id]['path'])
            image = cv2.imread(self.image_info[image_id]['path'])
            # If grayscale. Convert to RGB for consistency.
            if image.ndim != 3:
                image = cv2.COLOR_GRAY2RGB(image)
            # If has an alpha channel, remove it for consistency
            if image.shape[-1] == 4:
                image = image[..., :3]
            return image

        def load_mask(self, image_id):
            """Load instance masks for the given image.

            Different datasets use different ways to store masks. Override this
            method to load instance masks and return them in the form of am
            array of binary masks of shape [height, width, instances].

            Returns:
                masks: A bool array of shape [height, width, instance count] with
                    a binary mask per instance.
                class_ids: a 1D array of class IDs of the instance masks.
            """
            # Override this function to load a mask from your dataset.
            # Otherwise, it returns an empty mask.
            logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
            mask = np.empty([0, 0, 0])
            class_ids = np.empty([0], np.int32)
            return mask, class_ids

    class Dataset(Dataset):
        """ class is adapted from the NucleusDataset class in the Matterport implementation."""
        def load_data(self, dataset_dir, subset, classname):
            """Load a subset of the nuclei dataset.

            dataset_dir: Root directory of the dataset
            subset: Subset to load. Either the name of the sub-directory,
                    such as stage1_train, stage1_test, ...etc. or, one of:
                    * train: stage1_train excluding validation images
                    * val: validation images from VAL_IMAGE_IDS
            """
            # Add classes. We have one class.
            # Naming the dataset classname
            self.add_class(classname, 1, classname)

            # Which subset?
            # "val": use hard-coded list above
            # "train": use data from stage1_train minus the hard-coded list above
            # else: use the data from the specified sub-directory
            assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
            dataset_dir = os.path.join(dataset_dir)
            if subset == "val":
                image_ids = VAL_IMAGE_IDS
            else:
                # Get image ids from directory names
                image_ids = next(os.walk(dataset_dir))[1]
                # image_ids = os.listdir(dataset_dir)
                if subset == "train":
                    image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

            # Add images
            for image_id in image_ids:
                self.add_image(
                    classname,
                    image_id=image_id,
                    # path=os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))
                    path=os.path.join(dataset_dir, "{}.png".format(image_id)))

        def load_mask(self, image_id):
            """Generate instance masks for an image.
           Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
            """
            info = self.image_info[image_id]
            # Get mask directory from image path
            mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

            # Read mask files from .png image
            mask = []
            for f in next(os.walk(mask_dir))[2]:
                if f.endswith(".png"):
                    m = cv2.imread(os.path.join(mask_dir, f)).astype(np.bool)
                    mask.append(m)
            mask = np.stack(mask, axis=-1)
            # Return mask, and array of class IDs of each instance. Since we have
            # one class ID, we return an array of ones
            return mask, np.ones([mask.shape[-1]], dtype=np.int32)

        def image_reference(self, image_id):
            """Return the path of the image."""
            info = self.image_info[image_id]
            if info["source"] == "nucleus":
                return info["id"]
            else:
                super(self.__class__, self).image_reference(image_id)

    def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
        """Resizes an image keeping the aspect ratio unchanged.

        min_dim: if provided, resizes the image such that it's smaller
            dimension == min_dim
        max_dim: if provided, ensures that the image longest side doesn't
            exceed this value.
        min_scale: if provided, ensure that the image is scaled up by at least
            this percent even if min_dim doesn't require it.
        mode: Resizing mode.
            none: No resizing. Return the image unchanged.
            square: Resize and pad with zeros to get a square image
                of size [max_dim, max_dim].
            pad64: Pads width and height with zeros to make them multiples of 64.
                   If min_dim or min_scale are provided, it scales the image up
                   before padding. max_dim is ignored in this mode.
                   The multiple of 64 is needed to ensure smooth scaling of feature
                   maps up and down the 6 levels of the FPN pyramid (2**6=64).
            crop: Picks random crops from the image. First, scales the image based
                  on min_dim and min_scale, then picks a random crop of
                  size min_dim x min_dim. Can be used in training only.
                  max_dim is not used in this mode.

        Returns:
        image: the resized image
        window: (y1, x1, y2, x2). If max_dim is provided, padding might
            be inserted in the returned image. If so, this window is the
            coordinates of the image part of the full image (excluding
            the padding). The x2, y2 pixels are not included.
        scale: The scale factor used to resize the image
        padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
        """
        # Keep track of image dtype and return results in the same dtype
        image_dtype = image.dtype
        # Default window (y1, x1, y2, x2) and default scale == 1.
        h, w = image.shape[:2]
        window = (0, 0, h, w)
        scale = 1
        padding = [(0, 0), (0, 0), (0, 0)]
        crop = None

        if mode == "none":
            return image, window, scale, padding, crop

        # Scale?
        if min_dim:
            # Scale up but not down
            scale = max(1, min_dim / min(h, w))
        if min_scale and scale < min_scale:
            scale = min_scale

        # Does it exceed max dim?
        if max_dim and mode == "square":
            image_max = max(h, w)
            if round(image_max * scale) > max_dim:
                scale = max_dim / image_max

        # Resize image using bilinear interpolation
        if scale != 1:
            image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)

        # Need padding or cropping?
        if mode == "square":
            # Get new height and width
            h, w = image.shape[:2]
            top_pad = (max_dim - h) // 2
            bottom_pad = max_dim - h - top_pad
            left_pad = (max_dim - w) // 2
            right_pad = max_dim - w - left_pad
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        elif mode == "pad64":
            h, w = image.shape[:2]
            # Both sides must be divisible by 64
            if min_dim:  # condition when min_dim exists
                assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
            # Height
            if h % 64 > 0:
                max_h = h - (h % 64) + 64
                top_pad = (max_h - h) // 2
                bottom_pad = max_h - h - top_pad
            else:
                top_pad = bottom_pad = 0
            # Width
            if w % 64 > 0:
                max_w = w - (w % 64) + 64
                left_pad = (max_w - w) // 2
                right_pad = max_w - w - left_pad
            else:
                left_pad = right_pad = 0
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        elif mode == "crop":
            # Pick a random crop
            h, w = image.shape[:2]
            y = random.randint(0, (h - min_dim))
            x = random.randint(0, (w - min_dim))
            crop = (y, x, min_dim, min_dim)
            image = image[y:y + min_dim, x:x + min_dim]
            window = (0, 0, min_dim, min_dim)
        else:
            raise Exception("Mode {} not supported".format(mode))
        return image.astype(image_dtype), window, scale, padding, crop

    def unmold_mask(mask, bbox, image_shape):
        """Converts a mask generated by the neural network to a format similar
        to its original shape.
        mask: [height, width] of type float. A small, typically 28x28 mask.
        bbox: [y1, x1, y2, x2]. The box to fit the mask in.

        Returns a binary mask with the same size as the original image.
        """
        threshold = 0.5
        y1, x1, y2, x2 = bbox
        mask = cv2.resize(mask, (x2 - x1, y2 - y1))
        mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

        # Put the mask in the right location.
        full_mask = np.zeros(image_shape[:2], dtype=np.bool)
        full_mask[y1:y2, x1:x2] = mask
        return full_mask

    ############################################################
    #  Configurations
    ############################################################

    class Configuration(Config):
        """Configuration for training on the nucleus segmentation dataset."""
        # Give the configuration a recognizable name
        NAME = "cell"

        # Adjust depending on your GPU memory
        IMAGES_PER_GPU = 6

        # Number of classes (including background)
        NUM_CLASSES = 1 + 1  # Background + nucleus

        # Number of training and validation steps per epoch
        STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
        VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

        # Don't exclude based on confidence. Since we have two classes
        # then 0.5 is the minimum anyway as it picks between nucleus and BG
        DETECTION_MIN_CONFIDENCE = 0

        # Backbone network architecture
        # Supported values are: resnet50, resnet101
        BACKBONE = "resnet101"

        # Input image resizing
        # Random crops of size 512x512
        IMAGE_RESIZE_MODE = "crop"
        IMAGE_MIN_DIM = 512
        IMAGE_MAX_DIM = 512
        IMAGE_MIN_SCALE = 2.0

        # Length of square anchor side in pixels
        RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

        # ROIs kept after non-maximum supression (training and inference)
        POST_NMS_ROIS_TRAINING = 1000
        POST_NMS_ROIS_INFERENCE = 2000

        # Non-max suppression threshold to filter RPN proposals.
        # You can increase this during training to generate more propsals.
        RPN_NMS_THRESHOLD = 0.9

        # How many anchors per image to use for RPN training
        RPN_TRAIN_ANCHORS_PER_IMAGE = 64

        # Image mean (RGB)
        MEAN_PIXEL = np.array([36.97, 19.97, 36.97])

        # If enabled, resizes instance masks to a smaller size to reduce
        # memory load. Recommended when using high-resolution images.
        USE_MINI_MASK = True
        MINI_MASK_SHAPE = (76, 76)  # (height, width) of the mini-mask

        # Number of ROIs per image to feed to classifier/mask heads
        # The Mask RCNN paper uses 512 but often the RPN doesn't generate
        # enough positive proposals to fill this and keep a positive:negative
        # ratio of 1:3. You can increase the number of proposals by adjusting
        # the RPN NMS threshold.
        TRAIN_ROIS_PER_IMAGE = 128

        # Maximum number of ground truth instances to use in one image
        MAX_GT_INSTANCES = 400

        # Max number of final detections per image
        # this is computed based on input image size
        # we calculated it as 600 per image size of 512 by 512
        DETECTION_MAX_INSTANCES = math.ceil((HightOfImg/512)) * math.ceil((WidthOfImg/512)) * 600

    class InferenceConfig(Configuration):
        # Set batch size to 1 to run one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # Don't resize image for inferencing
        IMAGE_RESIZE_MODE = "pad64"
        # Non-max suppression threshold to filter RPN proposals.
        # You can increase this during training to generate more propsals.
        RPN_NMS_THRESHOLD = 0.7

    ############################################################
    #  Anchors
    ############################################################

    def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
        """
        scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
        ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
        shape: [height, width] spatial shape of the feature map over which
                to generate anchors.
        feature_stride: Stride of the feature map relative to the image in pixels.
        anchor_stride: Stride of anchors on the feature map. For example, if the
            value is 2 then generate anchors for every other feature map pixel.
        """
        # Get all combinations of scales and ratios
        scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
        scales = scales.flatten()
        ratios = ratios.flatten()

        # Enumerate heights and widths from scales and ratios
        heights = scales / np.sqrt(ratios)
        widths = scales * np.sqrt(ratios)

        # Enumerate shifts in feature space
        shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
        shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = np.stack(
            [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
        box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

        # Convert to corner coordinates (y1, x1, y2, x2)
        boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                                box_centers + 0.5 * box_sizes], axis=1)
        return boxes

    def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                                 anchor_stride):
        """Generate anchors at different levels of a feature pyramid. Each scale
        is associated with a level of the pyramid, but each ratio is used in
        all levels of the pyramid.

        Returns:
        anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
            with the same order of the given scales. So, anchors of scale[0] come
            first, then anchors of scale[1], and so on.
        """
        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        anchors = []
        for i in range(len(scales)):
            anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                            feature_strides[i], anchor_stride))
        return np.concatenate(anchors, axis=0)

    ############################################################
    #  Batch Slicing
    ############################################################

    def batch_slice(inputs, graph_fn, batch_size, names=None):
        """Splits inputs into slices and feeds each slice to a copy of the given
        computation graph and then combines the results. It allows you to run a
        graph on a batch of inputs even if the graph is written to support one
        instance only.

        inputs: list of tensors. All must have the same first dimension length
        graph_fn: A function that returns a TF tensor that's part of a graph.
        batch_size: number of slices to divide the data into.
        names: If provided, assigns names to the resulting tensors.
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        outputs = []
        for i in range(batch_size):
            inputs_slice = [x[i] for x in inputs]
            output_slice = graph_fn(*inputs_slice)
            if not isinstance(output_slice, (tuple, list)):
                output_slice = [output_slice]
            outputs.append(output_slice)
        # Change outputs from a list of slices where each is
        # a list of outputs to a list of outputs and each has
        # a list of slices
        outputs = list(zip(*outputs))

        if names is None:
            names = [None] * len(outputs)

        result = [tf.stack(o, axis=0, name=n)
                  for o, n in zip(outputs, names)]
        if len(result) == 1:
            result = result[0]

        return result

    def norm_boxes(boxes, shape):
        """Converts boxes from pixel coordinates to normalized coordinates.
        boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
        shape: [..., (height, width)] in pixels

        Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
        coordinates it's inside the box.

        Returns:
            [N, (y1, x1, y2, x2)] in normalized coordinates
        """
        h, w = shape
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.divide((boxes - shift), scale).astype(np.float32)

    def denorm_boxes(boxes, shape):
        """Converts boxes from normalized coordinates to pixel coordinates.
        boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
        shape: [..., (height, width)] in pixels

        Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
        coordinates it's inside the box.

        Returns:
            [N, (y1, x1, y2, x2)] in pixel coordinates
        """
        h, w = shape
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)

    ############################################################
    #  Utility Functions
    ############################################################

    def log(text, array=None):
        """Prints a text message. And, optionally, if a Numpy array is provided it
        prints it's shape, min, and max values.
        """
        if array is not None:
            text = text.ljust(25)
            text += ("shape: {:20}  ".format(str(array.shape)))
            if array.size:
                text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(), array.max()))
            else:
                text += ("min: {:10}  max: {:10}".format("", ""))
            text += "  {}".format(array.dtype)
        print(text)

    class BatchNorm(KL.BatchNormalization):
        """Extends the Keras BatchNormalization class to allow a central place
        to make changes if needed.

        Batch normalization has a negative effect on training if batches are small
        so this layer is often frozen (via setting in Config class) and functions
        as linear layer.
        """

        def call(self, inputs, training=None):
            """
            Note about training values:
                None: Train BN layers. This is the normal mode
                False: Freeze BN layers. Good when batch size is small
                True: (don't use). Set layer in training mode even when making inferences
            """
            return super(self.__class__, self).call(inputs, training=training)

    def compute_backbone_shapes(config, image_shape):
        """Computes the width and height of each stage of the backbone network.

        Returns:
            [N, (height, width)]. Where N is the number of stages
        """
        if callable(config.BACKBONE):
            return config.COMPUTE_BACKBONE_SHAPE(image_shape)

        # Currently supports ResNet only
        assert config.BACKBONE in ["resnet50", "resnet101"]
        return np.array(
            [[int(math.ceil(image_shape[0] / stride)),
              int(math.ceil(image_shape[1] / stride))]
             for stride in config.BACKBONE_STRIDES])

    ############################################################
    #  Resnet Graph
    ############################################################

    def identity_block(input_tensor, kernel_size, filters, stage, block,
                       use_bias=True, train_bn=True):
        """The identity_block is the block that has no conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            use_bias: Boolean. To use or not use a bias in conv layers.
            train_bn: Boolean. Train or freeze Batch Norm layers
        """
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                      use_bias=use_bias)(input_tensor)
        x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias)(x)
        x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                      use_bias=use_bias)(x)
        x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

        x = KL.Add()([x, input_tensor])
        x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x

    def conv_block(input_tensor, kernel_size, filters, stage, block,
                   strides=(2, 2), use_bias=True, train_bn=True):
        """conv_block is the block that has a conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            use_bias: Boolean. To use or not use a bias in conv layers.
            train_bn: Boolean. Train or freeze Batch Norm layers
        Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
        And the shortcut should have subsample=(2,2) as well
        """
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
        x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias)(x)
        x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                                               '2c', use_bias=use_bias)(x)
        x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

        shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
        shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

        x = KL.Add()([x, shortcut])
        x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x

    def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
        """Build a ResNet graph.
            architecture: Can be resnet50 or resnet101
            stage5: Boolean. If False, stage5 of the network is not created
            train_bn: Boolean. Train or freeze Batch Norm layers
        """
        assert architecture in ["resnet50", "resnet101"]
        # Stage 1
        x = KL.ZeroPadding2D((3, 3))(input_image)
        x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
        x = BatchNorm(name='bn_conv1')(x, training=train_bn)
        x = KL.Activation('relu')(x)
        C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        # Stage 2
        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
        C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
        # Stage 3
        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
        C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
        # Stage 4
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
        block_count = {"resnet50": 5, "resnet101": 22}[architecture]
        for i in range(block_count):
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
        C4 = x
        # Stage 5
        if stage5:
            x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
            x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
            C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
        else:
            C5 = None
        return [C1, C2, C3, C4, C5]

    ############################################################
    #  Proposal Layer
    ############################################################

    def apply_box_deltas_graph(boxes, deltas):
        """Applies the given deltas to the given boxes.
        boxes: [N, (y1, x1, y2, x2)] boxes to update
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        """
        # Convert to y, x, h, w
        height = boxes[:, 2] - boxes[:, 0]
        width = boxes[:, 3] - boxes[:, 1]
        center_y = boxes[:, 0] + 0.5 * height
        center_x = boxes[:, 1] + 0.5 * width
        # Apply deltas
        center_y += deltas[:, 0] * height
        center_x += deltas[:, 1] * width
        height *= tf.exp(deltas[:, 2])
        width *= tf.exp(deltas[:, 3])
        # Convert back to y1, x1, y2, x2
        y1 = center_y - 0.5 * height
        x1 = center_x - 0.5 * width
        y2 = y1 + height
        x2 = x1 + width
        result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
        return result

    def clip_boxes_graph(boxes, window):
        """
        boxes: [N, (y1, x1, y2, x2)]
        window: [4] in the form y1, x1, y2, x2
        """
        # Split
        wy1, wx1, wy2, wx2 = tf.split(window, 4)
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
        # Clip
        y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
        x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
        y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
        x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
        clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
        clipped.set_shape((clipped.shape[0], 4))
        return clipped

    class ProposalLayer(KE.Layer):
        """Receives anchor scores and selects a subset to pass as proposals
        to the second stage. Filtering is done based on anchor scores and
        non-max suppression to remove overlaps. It also applies bounding
        box refinement deltas to anchors.

        Inputs:
            rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
            rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
            anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

        Returns:
            Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
        """

        def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
            super(ProposalLayer, self).__init__(**kwargs)
            self.config = config
            self.proposal_count = proposal_count
            self.nms_threshold = nms_threshold

        def call(self, inputs):
            # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
            scores = inputs[0][:, :, 1]
            # Box deltas [batch, num_rois, 4]
            deltas = inputs[1]
            deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
            # Anchors
            anchors = inputs[2]

            # Improve performance by trimming to top anchors by score
            # and doing the rest on the smaller subset.
            pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
            ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                             name="top_anchors").indices
            scores = batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                       self.config.IMAGES_PER_GPU)
            deltas = batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                       self.config.IMAGES_PER_GPU)
            pre_nms_anchors = batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                                self.config.IMAGES_PER_GPU,
                                                names=["pre_nms_anchors"])

            # Apply deltas to anchors to get refined anchors.
            # [batch, N, (y1, x1, y2, x2)]
            boxes = batch_slice([pre_nms_anchors, deltas],
                                      lambda x, y: apply_box_deltas_graph(x, y),
                                      self.config.IMAGES_PER_GPU,
                                      names=["refined_anchors"])

            # Clip to image boundaries. Since we're in normalized coordinates,
            # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
            window = np.array([0, 0, 1, 1], dtype=np.float32)
            boxes = batch_slice(boxes,
                                      lambda x: clip_boxes_graph(x, window),
                                      self.config.IMAGES_PER_GPU,
                                      names=["refined_anchors_clipped"])

            # Filter out small boxes
            # According to Xinlei Chen's paper, this reduces detection accuracy
            # for small objects, so we're skipping it.

            # Non-max suppression
            def nms(boxes, scores):
                indices = tf.image.non_max_suppression(
                    boxes, scores, self.proposal_count,
                    self.nms_threshold, name="rpn_non_max_suppression")
                proposals = tf.gather(boxes, indices)
                # Pad if needed
                padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
                proposals = tf.pad(proposals, [(0, padding), (0, 0)])
                return proposals

            proposals = batch_slice([boxes, scores], nms,
                                          self.config.IMAGES_PER_GPU)
            return proposals

        def compute_output_shape(self, input_shape):
            return (None, self.proposal_count, 4)

    ############################################################
    #  ROIAlign Layer
    ############################################################

    def log2_graph(x):
        """Implementation of Log2. TF doesn't have a native implementation."""
        return tf.log(x) / tf.log(2.0)

    class PyramidROIAlign(KE.Layer):
        """Implements ROI Pooling on multiple levels of the feature pyramid.

        Params:
        - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

        Inputs:
        - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                 coordinates. Possibly padded with zeros if not enough
                 boxes to fill the array.
        - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        - feature_maps: List of feature maps from different levels of the pyramid.
                        Each is [batch, height, width, channels]

        Output:
        Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
        The width and height are those specific in the pool_shape in the layer
        constructor.
        """

        def __init__(self, pool_shape, **kwargs):
            super(PyramidROIAlign, self).__init__(**kwargs)
            self.pool_shape = tuple(pool_shape)

        def call(self, inputs):
            # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
            boxes = inputs[0]

            # Image meta
            # Holds details about the image. See compose_image_meta()
            image_meta = inputs[1]

            # Feature Maps. List of feature maps from different level of the
            # feature pyramid. Each is [batch, height, width, channels]
            feature_maps = inputs[2:]

            # Assign each ROI to a level in the pyramid based on the ROI area.
            y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
            h = y2 - y1
            w = x2 - x1
            # Use shape of first image. Images in a batch must have the same size.
            image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
            # Equation 1 in the Feature Pyramid Networks paper. Account for
            # the fact that our coordinates are normalized here.
            # e.g. a 224x224 ROI (in pixels) maps to P4
            image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
            roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
            roi_level = tf.minimum(5, tf.maximum(
                2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
            roi_level = tf.squeeze(roi_level, 2)

            # Loop through levels and apply ROI pooling to each. P2 to P5.
            pooled = []
            box_to_level = []
            for i, level in enumerate(range(2, 6)):
                ix = tf.where(tf.equal(roi_level, level))
                level_boxes = tf.gather_nd(boxes, ix)

                # Box indices for crop_and_resize.
                box_indices = tf.cast(ix[:, 0], tf.int32)

                # Keep track of which box is mapped to which level
                box_to_level.append(ix)

                # Stop gradient propogation to ROI proposals
                level_boxes = tf.stop_gradient(level_boxes)
                box_indices = tf.stop_gradient(box_indices)

                # Crop and Resize
                # From Mask R-CNN paper: "We sample four regular locations, so
                # that we can evaluate either max or average pooling. In fact,
                # interpolating only a single value at each bin center (without
                # pooling) is nearly as effective."
                #
                # Here we use the simplified approach of a single value per bin,
                # which is how it's done in tf.crop_and_resize()
                # Result: [batch * num_boxes, pool_height, pool_width, channels]
                pooled.append(tf.image.crop_and_resize(
                    feature_maps[i], level_boxes, box_indices, self.pool_shape,
                    method="bilinear"))

            # Pack pooled features into one tensor
            pooled = tf.concat(pooled, axis=0)

            # Pack box_to_level mapping into one array and add another
            # column representing the order of pooled boxes
            box_to_level = tf.concat(box_to_level, axis=0)
            box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
            box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                     axis=1)

            # Rearrange pooled features to match the order of the original boxes
            # Sort box_to_level by batch then box index
            # TF doesn't have a way to sort by two columns, so merge them and sort.
            sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
            ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
                box_to_level)[0]).indices[::-1]
            ix = tf.gather(box_to_level[:, 2], ix)
            pooled = tf.gather(pooled, ix)

            # Re-add the batch dimension
            shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
            pooled = tf.reshape(pooled, shape)
            return pooled

        def compute_output_shape(self, input_shape):
            return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)

    ############################################################
    #  Detection Target Layer
    ############################################################

    def overlaps_graph(boxes1, boxes2):
        """Computes IoU overlaps between two sets of boxes.
        boxes1, boxes2: [N, (y1, x1, y2, x2)].
        """
        # 1. Tile boxes2 and repeat boxes1. This allows us to compare
        # every boxes1 against every boxes2 without loops.
        # TF doesn't have an equivalent to np.repeat() so simulate it
        # using tf.tile() and tf.reshape.
        b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                                [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
        b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
        # 2. Compute intersections
        b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
        b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
        y1 = tf.maximum(b1_y1, b2_y1)
        x1 = tf.maximum(b1_x1, b2_x1)
        y2 = tf.minimum(b1_y2, b2_y2)
        x2 = tf.minimum(b1_x2, b2_x2)
        intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
        # 3. Compute unions
        b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
        b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
        union = b1_area + b2_area - intersection
        # 4. Compute IoU and reshape to [boxes1, boxes2]
        iou = intersection / union
        overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
        return overlaps

    def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
        """Generates detection targets for one image. Subsamples proposals and
        generates target class IDs, bounding box deltas, and masks for each.

        Inputs:
        proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals.
        gt_class_ids: [MAX_GT_INSTANCES] int class IDs
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
        gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

        Returns: Target ROIs and corresponding class IDs, bounding box shifts,
        and masks.
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
        deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
        masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
               boundaries and resized to neural network output size.

        Note: Returned arrays might be zero padded if not enough target ROIs.
        """
        # Assertions
        asserts = [
            tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                      name="roi_assertion"),
        ]
        with tf.control_dependencies(asserts):
            proposals = tf.identity(proposals)

        # Remove zero padding
        proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
        gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                       name="trim_gt_class_ids")
        gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                             name="trim_gt_masks")

        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances. Exclude
        # them from training. A crowd box is given a negative class ID.
        crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
        non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
        crowd_boxes = tf.gather(gt_boxes, crowd_ix)
        gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
        gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
        gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

        # Compute overlaps matrix [proposals, gt_boxes]
        overlaps = overlaps_graph(proposals, gt_boxes)

        # Compute overlaps with crowd boxes [proposals, crowd_boxes]
        crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
        crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)

        # Determine positive and negative ROIs
        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
        positive_roi_bool = (roi_iou_max >= 0.5)
        positive_indices = tf.where(positive_roi_bool)[:, 0]
        # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
        negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                             config.ROI_POSITIVE_RATIO)
        positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
        # Gather selected ROIs
        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)

        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.cond(
            tf.greater(tf.shape(positive_overlaps)[1], 0),
            true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
            false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
        )
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

        # Compute bbox refinement for positive ROIs
        deltas = box_refinement_graph(positive_rois, roi_gt_boxes)
        deltas /= config.BBOX_STD_DEV

        # Assign positive ROIs to GT masks
        # Permute masks to [N, height, width, 1]
        transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
        # Pick the right mask for each ROI
        roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

        # Compute mask targets
        boxes = positive_rois
        if config.USE_MINI_MASK:
            # Transform ROI coordinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = tf.concat([y1, x1, y2, x2], 1)
        box_ids = tf.range(0, tf.shape(roi_masks)[0])
        masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                         box_ids,
                                         config.MASK_SHAPE)
        # Remove the extra dimension from masks.
        masks = tf.squeeze(masks, axis=3)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = tf.round(masks)

        # Append negative ROIs and pad bbox deltas and masks that
        # are not used for negative ROIs with zeros.
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        N = tf.shape(negative_rois)[0]
        P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
        rois = tf.pad(rois, [(0, P), (0, 0)])
        roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
        roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
        deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
        masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

        return rois, roi_gt_class_ids, deltas, masks

    class DetectionTargetLayer(KE.Layer):
        """Subsamples proposals and generates target box refinement, class_ids,
        and masks for each.

        Inputs:
        proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals.
        gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
        gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
                  coordinates.
        gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

        Returns: Target ROIs and corresponding class IDs, bounding box shifts,
        and masks.
        rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
              coordinates
        target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
        target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
        target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                     Masks cropped to bbox boundaries and resized to neural
                     network output size.

        Note: Returned arrays might be zero padded if not enough target ROIs.
        """

        def __init__(self, config, **kwargs):
            super(DetectionTargetLayer, self).__init__(**kwargs)
            self.config = config

        def call(self, inputs):
            proposals = inputs[0]
            gt_class_ids = inputs[1]
            gt_boxes = inputs[2]
            gt_masks = inputs[3]

            # Slice the batch and run a graph for each slice
            # TODO: Rename target_bbox to target_deltas for clarity
            names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
            outputs = batch_slice(
                [proposals, gt_class_ids, gt_boxes, gt_masks],
                lambda w, x, y, z: detection_targets_graph(
                    w, x, y, z, self.config),
                self.config.IMAGES_PER_GPU, names=names)
            return outputs

        def compute_output_shape(self, input_shape):
            return [
                (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
                (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
                (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
                (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
                 self.config.MASK_SHAPE[1])  # masks
            ]

        def compute_mask(self, inputs, mask=None):
            return [None, None, None, None]

    ############################################################
    #  Detection Layer
    ############################################################

    def refine_detections_graph(rois, probs, deltas, window, config):
        """Refine classified proposals and filter overlaps and return final
        detections.

        Inputs:
            rois: [N, (y1, x1, y2, x2)] in normalized coordinates
            probs: [N, num_classes]. Class probabilities.
            deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                    bounding box deltas.
            window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
                that contains the image excluding the padding.

        Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
            coordinates are normalized.
        """
        # Class IDs per ROI
        class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
        # Class probability of the top class of each ROI
        indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
        class_scores = tf.gather_nd(probs, indices)
        # Class-specific bounding box deltas
        deltas_specific = tf.gather_nd(deltas, indices)
        # Apply bounding box deltas
        # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
        refined_rois = apply_box_deltas_graph(
            rois, deltas_specific * config.BBOX_STD_DEV)
        # Clip boxes to image window
        refined_rois = clip_boxes_graph(refined_rois, window)

        # TODO: Filter out boxes with zero area

        # Filter out background boxes
        keep = tf.where(class_ids > 0)[:, 0]
        # Filter out low confidence boxes
        if config.DETECTION_MIN_CONFIDENCE:
            conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
            keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                            tf.expand_dims(conf_keep, 0))
            keep = tf.sparse_tensor_to_dense(keep)[0]

        # Apply per-class NMS
        # 1. Prepare variables
        pre_nms_class_ids = tf.gather(class_ids, keep)
        pre_nms_scores = tf.gather(class_scores, keep)
        pre_nms_rois = tf.gather(refined_rois, keep)
        unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

        def nms_keep_map(class_id):
            """Apply Non-Maximum Suppression on ROIs of the given class."""
            # Indices of ROIs of the given class
            ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
            # Apply NMS
            class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=config.DETECTION_MAX_INSTANCES,
                iou_threshold=config.DETECTION_NMS_THRESHOLD)
            # Map indices
            class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
            # Pad with -1 so returned tensors have the same shape
            gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
            class_keep = tf.pad(class_keep, [(0, gap)],
                                mode='CONSTANT', constant_values=-1)
            # Set shape so map_fn() can infer result shape
            class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
            return class_keep

        # 2. Map over class IDs
        nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                             dtype=tf.int64)
        # 3. Merge results into one list, and remove -1 padding
        nms_keep = tf.reshape(nms_keep, [-1])
        nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
        # 4. Compute intersection between keep and nms_keep
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(nms_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]
        # Keep top detections
        roi_count = config.DETECTION_MAX_INSTANCES
        class_scores_keep = tf.gather(class_scores, keep)
        num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
        top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
        keep = tf.gather(keep, top_ids)

        # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
        # Coordinates are normalized.
        detections = tf.concat([
            tf.gather(refined_rois, keep),
            tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

        # Pad with zeros if detections < DETECTION_MAX_INSTANCES
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
        detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
        return detections

    class DetectionLayer(KE.Layer):
        """Takes classified proposal boxes and their bounding box deltas and
        returns the final detection boxes.

        Returns:
        [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
        coordinates are normalized.
        """

        def __init__(self, config=None, **kwargs):
            super(DetectionLayer, self).__init__(**kwargs)
            self.config = config

        def call(self, inputs):
            rois = inputs[0]
            mrcnn_class = inputs[1]
            mrcnn_bbox = inputs[2]
            image_meta = inputs[3]

            # Get windows of images in normalized coordinates. Windows are the area
            # in the image that excludes the padding.
            # Use the shape of the first image in the batch to normalize the window
            # because we know that all images get resized to the same size.
            m = parse_image_meta_graph(image_meta)
            image_shape = m['image_shape'][0]
            window = norm_boxes_graph(m['window'], image_shape[:2])

            # Run detection refinement graph on each item in the batch
            detections_batch = batch_slice(
                [rois, mrcnn_class, mrcnn_bbox, window],
                lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
                self.config.IMAGES_PER_GPU)

            # Reshape output
            # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
            # normalized coordinates
            return tf.reshape(
                detections_batch,
                [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

        def compute_output_shape(self, input_shape):
            return (None, self.config.DETECTION_MAX_INSTANCES, 6)

    ############################################################
    #  Region Proposal Network (RPN)
    ############################################################

    def rpn_graph(feature_map, anchors_per_location, anchor_stride):
        """Builds the computation graph of Region Proposal Network.

        feature_map: backbone features [batch, height, width, depth]
        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                       every pixel in the feature map), or 2 (every other pixel).

        Returns:
            rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
            rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
            rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                      applied to anchors.
        """
        # TODO: check if stride of 2 causes alignment issues if the feature map
        # is not even.
        # Shared convolutional base of the RPN
        shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                           strides=anchor_stride,
                           name='rpn_conv_shared')(feature_map)

        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                      activation='linear', name='rpn_class_raw')(shared)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = KL.Lambda(
            lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

        # Softmax on last dimension of BG/FG.
        rpn_probs = KL.Activation(
            "softmax", name="rpn_class_xxx")(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                      activation='linear', name='rpn_bbox_pred')(shared)

        # Reshape to [batch, anchors, 4]
        rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

        return [rpn_class_logits, rpn_probs, rpn_bbox]

    def build_rpn_model(anchor_stride, anchors_per_location, depth):
        """Builds a Keras model of the Region Proposal Network.
        It wraps the RPN graph so it can be used multiple times with shared
        weights.

        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                       every pixel in the feature map), or 2 (every other pixel).
        depth: Depth of the backbone feature map.

        Returns a Keras Model object. The model outputs, when called, are:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                    applied to anchors.
        """
        input_feature_map = KL.Input(shape=[None, None, depth],
                                     name="input_rpn_feature_map")
        outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
        return KM.Model([input_feature_map], outputs, name="rpn_model")

    ############################################################
    #  Feature Pyramid Network Heads
    ############################################################

    def fpn_classifier_graph(rois, feature_maps, image_meta,
                             pool_size, num_classes, train_bn=True,
                             fc_layers_size=1024):
        """Builds the computation graph of the feature pyramid network classifier
        and regressor heads.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates.
        feature_maps: List of feature maps from different layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
        image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results
        train_bn: Boolean. Train or freeze Batch Norm layers
        fc_layers_size: Size of the 2 FC layers

        Returns:
            logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
            probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
            bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                         proposal boxes
        """
        # ROI Pooling
        # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
        x = PyramidROIAlign([pool_size, pool_size],
                            name="roi_align_classifier")([rois, image_meta] + feature_maps)
        # Two 1024 FC layers (implemented with Conv2D for consistency)
        x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                               name="mrcnn_class_conv1")(x)
        x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
        x = KL.Activation('relu')(x)
        x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
                               name="mrcnn_class_conv2")(x)
        x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                           name="pool_squeeze")(x)

        # Classifier head
        mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                                name='mrcnn_class_logits')(shared)
        mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                         name="mrcnn_class")(mrcnn_class_logits)

        # BBox head
        # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
        x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                               name='mrcnn_bbox_fc')(shared)
        # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
        s = K.int_shape(x)
        mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

        return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox

    def build_fpn_mask_graph(rois, feature_maps, image_meta,
                             pool_size, num_classes, train_bn=True):
        """Builds the computation graph of the mask head of Feature Pyramid Network.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates.
        feature_maps: List of feature maps from different layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
        image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results
        train_bn: Boolean. Train or freeze Batch Norm layers

        Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
        """
        # ROI Pooling
        # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
        x = PyramidROIAlign([pool_size, pool_size],
                            name="roi_align_mask")([rois, image_meta] + feature_maps)

        # Conv layers
        x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                               name="mrcnn_mask_conv1")(x)
        x = KL.TimeDistributed(BatchNorm(),
                               name='mrcnn_mask_bn1')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                               name="mrcnn_mask_conv2")(x)
        x = KL.TimeDistributed(BatchNorm(),
                               name='mrcnn_mask_bn2')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                               name="mrcnn_mask_conv3")(x)
        x = KL.TimeDistributed(BatchNorm(),
                               name='mrcnn_mask_bn3')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                               name="mrcnn_mask_conv4")(x)
        x = KL.TimeDistributed(BatchNorm(),
                               name='mrcnn_mask_bn4')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                               name="mrcnn_mask_deconv")(x)
        x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                               name="mrcnn_mask")(x)
        return x

    ############################################################
    #  Data Generator
    ############################################################

    def load_image_only(dataset, config, image_id, augment=False, augmentation=None,
                      use_mini_mask=False):
        """Load and return ground truth data for an image (image, mask, bounding boxes).

        augment: (deprecated. Use augmentation instead). If true, apply random
            image augmentation. Currently, only horizontal flipping is offered.
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
            For example, passing imgaug.augmenters.Fliplr(0.5) flips images
            right/left 50% of the time.
        use_mini_mask: If False, returns full-size masks that are the same height
            and width as the original image. These can be big, for example
            1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
            224x224 and are generated by extracting the bounding box of the
            object and resizing it to MINI_MASK_SHAPE.

        Returns:
        image: [height, width, 3]
        shape: the original shape of the image before resizing and cropping.
        class_ids: [instance_count] Integer class IDs
        bbox: [instance_count, (y1, x1, y2, x2)]
        mask: [height, width, instance_count]. The height and width are those
            of the image unless use_mini_mask is True, in which case they are
            defined in MINI_MASK_SHAPE.
        """
        # Load image and mask
        image = dataset.load_image(image_id)
        original_shape = image.shape
        image, window, scale, padding, crop = resize_image(
            image,
            min_scale=config.IMAGE_MIN_SCALE,
            mode=config.IMAGE_RESIZE_MODE)

        # Random horizontal flips.
        # TODO: will be removed in a future update in favor of augmentation
        if augment:
            logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
            if random.randint(0, 1):
                image = np.fliplr(image)

        # Augmentation
        # This requires the imgaug lib (https://github.com/aleju/imgaug)
        if augmentation:
            import imgaug

            # Augmenters that are safe to apply to masks
            # Some, such as Affine, have settings that make them unsafe, so always
            # test your augmentation on masks
            MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                               "Fliplr", "Flipud", "CropAndPad",
                               "Affine", "PiecewiseAffine"]

            def hook(images, augmenter, parents, default):
                """Determines which augmenters to apply to masks."""
                return augmenter.__class__.__name__ in MASK_AUGMENTERS

            # Store shapes before augmentation to compare
            image_shape = image.shape
            # Make augmenters deterministic to apply similarly to images and masks
            det = augmentation.to_deterministic()
            image = det.augment_image(image)
            # Verify that shapes didn't change)
            assert image.shape == image_shape, "Augmentation shouldn't change image size"



        # Active classes
        # Different datasets have different classes, so track the
        # classes supported in the dataset of this image.
        active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
        source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
        active_class_ids[source_class_ids] = 1

        # Image meta data
        image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                        window, scale, active_class_ids)

        return image, image_meta, padding

    ############################################################
    #  MaskRCNN Class
    ############################################################

    class MaskRCNN():
        """Encapsulates the Mask RCNN model functionality.

        The actual Keras model is in the keras_model property.
        """

        def __init__(self, mode, config):
            """
            mode: Either "training" or "inference"
            config: A Sub-class of the Config class
            model_dir: Directory to save training logs and trained weights
            """
            assert mode in ['training', 'inference']
            self.mode = mode
            self.config = config
            self.set_log_dir()
            self.keras_model = self.build(mode=mode, config=config)

        def build(self, mode, config):
            """Build Mask R-CNN architecture.
                input_shape: The shape of the input image.
                mode: Either "training" or "inference". The inputs and
                    outputs of the model differ accordingly.
            """
            assert mode in ['training', 'inference']

            # Image size must be dividable by 2 multiple times
            h, w = config.IMAGE_SHAPE[:2]
            if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
                raise Exception("Image size must be dividable by 2 at least 6 times "
                                "to avoid fractions when downscaling and upscaling."
                                "For example, use 256, 320, 384, 448, 512, ... etc. ")

            # Inputs
            input_image = KL.Input(
                shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
            input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                        name="input_image_meta")
            if mode == "training":
                # RPN GT
                input_rpn_match = KL.Input(
                    shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
                input_rpn_bbox = KL.Input(
                    shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

                # Detection GT (class IDs, bounding boxes, and masks)
                # 1. GT Class IDs (zero padded)
                input_gt_class_ids = KL.Input(
                    shape=[None], name="input_gt_class_ids", dtype=tf.int32)
                # 2. GT Boxes in pixels (zero padded)
                # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
                input_gt_boxes = KL.Input(
                    shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
                # Normalize coordinates
                gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_gt_boxes)
                # 3. GT Masks (zero padded)
                # [batch, height, width, MAX_GT_INSTANCES]
                if config.USE_MINI_MASK:
                    input_gt_masks = KL.Input(
                        shape=[config.MINI_MASK_SHAPE[0],
                               config.MINI_MASK_SHAPE[1], None],
                        name="input_gt_masks", dtype=bool)
                else:
                    input_gt_masks = KL.Input(
                        shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                        name="input_gt_masks", dtype=bool)
            elif mode == "inference":
                # Anchors in normalized coordinates
                input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

            # Build the shared convolutional layers.
            # Bottom-up Layers
            # Returns a list of the last layers of each stage, 5 in total.
            # Don't create the thead (stage 5), so we pick the 4th item in the list.
            if callable(config.BACKBONE):
                _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
                                                    train_bn=config.TRAIN_BN)
            else:
                _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
                                                 stage5=True, train_bn=config.TRAIN_BN)
            # Top-down Layers
            # TODO: add assert to varify feature map sizes match what's in config
            P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
            P4 = KL.Add(name="fpn_p4add")([
                KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
            P3 = KL.Add(name="fpn_p3add")([
                KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
            P2 = KL.Add(name="fpn_p2add")([
                KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
            # Attach 3x3 conv to all P layers to get the final feature maps.
            P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
            P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
            P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
            P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
            # P6 is used for the 5th anchor scale in RPN. Generated by
            # subsampling from P5 with stride of 2.
            P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

            # Note that P6 is used in RPN, but not in the classifier heads.
            rpn_feature_maps = [P2, P3, P4, P5, P6]
            mrcnn_feature_maps = [P2, P3, P4, P5]

            # Anchors
            if mode == "training":
                anchors = self.get_anchors(config.IMAGE_SHAPE)
                # Duplicate across the batch dimension because Keras requires it
                # TODO: can this be optimized to avoid duplicating the anchors?
                anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
                # A hack to get around Keras's bad support for constants
                anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
            else:
                anchors = input_anchors

            # RPN Model
            rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                                  len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
            # Loop through pyramid layers
            layer_outputs = []  # list of lists
            for p in rpn_feature_maps:
                layer_outputs.append(rpn([p]))
            # Concatenate layer outputs
            # Convert from list of lists of level outputs to list of lists
            # of outputs across levels.
            # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
            output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
            outputs = list(zip(*layer_outputs))
            outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                       for o, n in zip(outputs, output_names)]

            rpn_class_logits, rpn_class, rpn_bbox = outputs

            # Generate proposals
            # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
            # and zero padded.
            proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" \
                else config.POST_NMS_ROIS_INFERENCE
            rpn_rois = ProposalLayer(
                proposal_count=proposal_count,
                nms_threshold=config.RPN_NMS_THRESHOLD,
                name="ROI",
                config=config)([rpn_class, rpn_bbox, anchors])

            if mode == "training":
                # Class ID mask to mark class IDs supported by the dataset the image
                # came from.
                active_class_ids = KL.Lambda(
                    lambda x: parse_image_meta_graph(x)["active_class_ids"]
                )(input_image_meta)

                if not config.USE_RPN_ROIS:
                    # Ignore predicted ROIs and use ROIs provided as an input.
                    input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                          name="input_roi", dtype=np.int32)
                    # Normalize coordinates
                    target_rois = KL.Lambda(lambda x: norm_boxes_graph(
                        x, K.shape(input_image)[1:3]))(input_rois)
                else:
                    target_rois = rpn_rois

                # Generate detection targets
                # Subsamples proposals and generates target outputs for training
                # Note that proposal class IDs, gt_boxes, and gt_masks are zero
                # padded. Equally, returned rois and targets are zero padded.
                rois, target_class_ids, target_bbox, target_mask = \
                    DetectionTargetLayer(config, name="proposal_targets")([
                        target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

                # Network Heads
                # TODO: verify that this handles zero padded ROIs
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                    fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                         config.POOL_SIZE, config.NUM_CLASSES,
                                         train_bn=config.TRAIN_BN,
                                         fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

                mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                                  input_image_meta,
                                                  config.MASK_POOL_SIZE,
                                                  config.NUM_CLASSES,
                                                  train_bn=config.TRAIN_BN)

                # TODO: clean up (use tf.identify if necessary)
                output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

                # Losses
                rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                    [input_rpn_match, rpn_class_logits])
                rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                    [input_rpn_bbox, input_rpn_match, rpn_bbox])
                class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                    [target_class_ids, mrcnn_class_logits, active_class_ids])
                bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                    [target_bbox, target_class_ids, mrcnn_bbox])
                mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                    [target_mask, target_class_ids, mrcnn_mask])

                # Model
                inputs = [input_image, input_image_meta,
                          input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
                if not config.USE_RPN_ROIS:
                    inputs.append(input_rois)
                outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                           mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                           rpn_rois, output_rois,
                           rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
                model = KM.Model(inputs, outputs, name='mask_rcnn')
            else:
                # Network Heads
                # Proposal classifier and BBox regressor heads
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                    fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                         config.POOL_SIZE, config.NUM_CLASSES,
                                         train_bn=config.TRAIN_BN,
                                         fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

                # Detections
                # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
                # normalized coordinates
                detections = DetectionLayer(config, name="mrcnn_detection")(
                    [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

                # Create masks for detections
                detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
                mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                                  input_image_meta,
                                                  config.MASK_POOL_SIZE,
                                                  config.NUM_CLASSES,
                                                  train_bn=config.TRAIN_BN)

                model = KM.Model([input_image, input_image_meta, input_anchors],
                                 [detections, mrcnn_class, mrcnn_bbox,
                                  mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                                 name='mask_rcnn')

            # Add multi-GPU support.
            if config.GPU_COUNT > 1:
                from mrcnn.parallel_model import ParallelModel
                model = ParallelModel(model, config.GPU_COUNT)

            return model

        def load_weights(self, filepath, by_name=False, exclude=None):
            """Modified version of the corresponding Keras function with
            the addition of multi-GPU support and the ability to exclude
            some layers from loading.
            exclude: list of layer names to exclude
            """
            import h5py
            # Conditional import to support versions of Keras before 2.2
            # TODO: remove in about 6 months (end of 2018)
            try:
                from keras.engine import saving
            except ImportError:
                # Keras before 2.2 used the 'topology' namespace.
                from keras.engine import topology as saving

            if exclude:
                by_name = True

            if h5py is None:
                raise ImportError('`load_weights` requires h5py.')
            f = h5py.File(filepath, mode='r')
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']

            # In multi-GPU training, we wrap the model. Get layers
            # of the inner model because they have the weights.
            keras_model = self.keras_model
            layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
                else keras_model.layers

            # Exclude some layers
            if exclude:
                layers = filter(lambda l: l.name not in exclude, layers)

            if by_name:
                saving.load_weights_from_hdf5_group_by_name(f, layers)
            else:
                saving.load_weights_from_hdf5_group(f, layers)
            if hasattr(f, 'close'):
                f.close()

            # Update the log directory
            self.set_log_dir(filepath)

        def mold_inputs(self, images):
            """Takes a list of images and modifies them to the format expected
            as an input to the neural network.
            images: List of image matrices [height,width,depth]. Images can have
                different sizes.

            Returns 3 Numpy matrices:
            molded_images: [N, h, w, 3]. Images resized and normalized.
            image_metas: [N, length of meta data]. Details about each image.
            windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
                original image (padding excluded).
            """
            molded_images = []
            image_metas = []
            windows = []
            for image in images:
                # Resize image
                # TODO: move resizing to mold_image()
                molded_image, window, scale, padding, crop = resize_image(
                    image,
                    min_scale=self.config.IMAGE_MIN_SCALE,
                    mode=self.config.IMAGE_RESIZE_MODE)
                molded_image = mold_image(molded_image, self.config)
                # Build image_meta
                image_meta = compose_image_meta(
                    0, image.shape, molded_image.shape, window, scale,
                    np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
                # Append
                molded_images.append(molded_image)
                windows.append(window)
                image_metas.append(image_meta)
            # Pack into arrays
            molded_images = np.stack(molded_images)
            image_metas = np.stack(image_metas)
            windows = np.stack(windows)
            return molded_images, image_metas, windows

        def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                              image_shape, window):
            """Reformats the detections of one image from the format of the neural
            network output to a format suitable for use in the rest of the
            application.

            detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
            mrcnn_mask: [N, height, width, num_classes]
            original_image_shape: [H, W, C] Original image shape before resizing
            image_shape: [H, W, C] Shape of the image after resizing and padding
            window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                    image is excluding the padding.

            Returns:
            boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
            class_ids: [N] Integer class IDs for each bounding box
            scores: [N] Float probability scores of the class_id
            masks: [height, width, num_instances] Instance masks
            """
            # How many detections do we have?
            # Detections array is padded with zeros. Find the first class_id == 0.
            zero_ix = np.where(detections[:, 4] == 0)[0]
            N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

            # Extract boxes, class_ids, scores, and class-specific masks
            boxes = detections[:N, :4]
            class_ids = detections[:N, 4].astype(np.int32)
            scores = detections[:N, 5]
            masks = mrcnn_mask[np.arange(N), :, :, class_ids]

            # Translate normalized coordinates in the resized image to pixel
            # coordinates in the original image before resizing
            window = norm_boxes(window, image_shape[:2])
            wy1, wx1, wy2, wx2 = window
            shift = np.array([wy1, wx1, wy1, wx1])
            wh = wy2 - wy1  # window height
            ww = wx2 - wx1  # window width
            scale = np.array([wh, ww, wh, ww])
            # Convert boxes to normalized coordinates on the window
            boxes = np.divide(boxes - shift, scale)
            # Convert boxes to pixel coordinates on the original image
            boxes = denorm_boxes(boxes, original_image_shape[:2])

            # Filter out detections with zero area. Happens in early training when
            # network weights are still random
            exclude_ix = np.where(
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
            if exclude_ix.shape[0] > 0:
                boxes = np.delete(boxes, exclude_ix, axis=0)
                class_ids = np.delete(class_ids, exclude_ix, axis=0)
                scores = np.delete(scores, exclude_ix, axis=0)
                masks = np.delete(masks, exclude_ix, axis=0)
                N = class_ids.shape[0]

            # Resize masks to original image size and set boundary threshold.
            full_masks = []
            for i in range(N):
                # Convert neural network mask to full size mask
                full_mask = unmold_mask(masks[i], boxes[i], original_image_shape)
                full_masks.append(full_mask)
            full_masks = np.stack(full_masks, axis=-1) \
                if full_masks else np.empty(original_image_shape[:2] + (0,))

            return boxes, class_ids, scores, full_masks

        def detect(self, images, verbose=0):
            """Runs the detection pipeline.

            images: List of images, potentially of different sizes.

            Returns a list of dicts, one dict per image. The dict contains:
            rois: [N, (y1, x1, y2, x2)] detection bounding boxes
            class_ids: [N] int class IDs
            scores: [N] float probability scores for the class IDs
            masks: [H, W, N] instance binary masks
            """
            assert self.mode == "inference", "Create model in inference mode."
            assert len(
                images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

            if verbose:
                log("Processing {} images".format(len(images)))
                for image in images:
                    log("image", image)

            # Mold inputs to format expected by the neural network
            molded_images, image_metas, windows = self.mold_inputs(images)

            # Validate image sizes
            # All images in a batch MUST be of the same size
            image_shape = molded_images[0].shape
            for g in molded_images[1:]:
                assert g.shape == image_shape, \
                    "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

            # Anchors
            anchors = self.get_anchors(image_shape)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

            if verbose:
                log("molded_images", molded_images)
                log("image_metas", image_metas)
                log("anchors", anchors)
            # Run object detection
            detections, _, _, mrcnn_mask, _, _, _ = \
                self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
            # Process detections
            results = []
            for i, image in enumerate(images):
                final_rois, final_class_ids, final_scores, final_masks = \
                    self.unmold_detections(detections[i], mrcnn_mask[i],
                                           image.shape, molded_images[i].shape,
                                           windows[i])
                results.append({
                    "rois": final_rois,
                    "class_ids": final_class_ids,
                    "scores": final_scores,
                    "masks": final_masks,
                })
            return results

        def detect_molded(self, molded_images, image_metas, verbose=0):
            """Runs the detection pipeline, but expect inputs that are
            molded already. Used mostly for debugging and inspecting
            the model.

            molded_images: List of images loaded using load_image_gt()
            image_metas: image meta data, also returned by load_image_gt()

            Returns a list of dicts, one dict per image. The dict contains:
            rois: [N, (y1, x1, y2, x2)] detection bounding boxes
            class_ids: [N] int class IDs
            scores: [N] float probability scores for the class IDs
            masks: [H, W, N] instance binary masks
            """
            assert self.mode == "inference", "Create model in inference mode."
            assert len(molded_images) == self.config.BATCH_SIZE, \
                "Number of images must be equal to BATCH_SIZE"

            if verbose:
                log("Processing {} images".format(len(molded_images)))
                for image in molded_images:
                    log("image", image)

            # Validate image sizes
            # All images in a batch MUST be of the same size
            image_shape = molded_images[0].shape
            for g in molded_images[1:]:
                assert g.shape == image_shape, "Images must have the same size"

            # Anchors
            anchors = self.get_anchors(image_shape)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

            if verbose:
                log("molded_images", molded_images)
                log("image_metas", image_metas)
                log("anchors", anchors)
            # Run object detection
            detections, _, _, mrcnn_mask, _, _, _ = \
                self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
            # Process detections
            results = []
            for i, image in enumerate(molded_images):
                window = [0, 0, image.shape[0], image.shape[1]]
                final_rois, final_class_ids, final_scores, final_masks = \
                    self.unmold_detections(detections[i], mrcnn_mask[i],
                                           image.shape, molded_images[i].shape,
                                           window)
                results.append({
                    "rois": final_rois,
                    "class_ids": final_class_ids,
                    "scores": final_scores,
                    "masks": final_masks,
                })
            return results

        def set_log_dir(self, model_path=None):
            """Sets the model log directory and epoch counter.

            model_path: If None, or a format different from what this code uses
                then set a new log directory and start epochs from 0. Otherwise,
                extract the log directory and the epoch counter from the file
                name.
            """
            # Set date and epoch counter as if starting a new model
            self.epoch = 0
            now = datetime.datetime.now()

            # If we have a model path with date and epochs use them
            if model_path:
                # Continue from we left of. Get epoch and date from the file name
                # A sample model path might look like:
                # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
                # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
                regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
                m = re.match(regex, model_path)
                if m:
                    now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                            int(m.group(4)), int(m.group(5)))
                    # Epoch number in file is 1-based, and in Keras code it's 0-based.
                    # So, adjust for that then increment by one to start from the next epoch
                    self.epoch = int(m.group(6)) - 1 + 1
                    print('Re-starting from epoch %d' % self.epoch)

        def get_anchors(self, image_shape):
            """Returns anchor pyramid for the given image size."""
            backbone_shapes = compute_backbone_shapes(self.config, image_shape)
            # Cache anchors and reuse if image shape is the same
            if not hasattr(self, "_anchor_cache"):
                self._anchor_cache = {}
            if not tuple(image_shape) in self._anchor_cache:
                # Generate Anchors
                a = generate_pyramid_anchors(
                    self.config.RPN_ANCHOR_SCALES,
                    self.config.RPN_ANCHOR_RATIOS,
                    backbone_shapes,
                    self.config.BACKBONE_STRIDES,
                    self.config.RPN_ANCHOR_STRIDE)
                # Keep a copy of the latest anchors in pixel coordinates because
                # it's used in inspect_model notebooks.
                # TODO: Remove this after the notebook are refactored to not use it
                self.anchors = a
                # Normalize coordinates
                self._anchor_cache[tuple(image_shape)] = norm_boxes(a, image_shape[:2])
            return self._anchor_cache[tuple(image_shape)]

    ############################################################
    #  Data Formatting
    ############################################################

    def compose_image_meta(image_id, original_image_shape, image_shape,
                           window, scale, active_class_ids):
        """Takes attributes of an image and puts them in one 1D array.

        image_id: An int ID of the image. Useful for debugging.
        original_image_shape: [H, W, C] before resizing or padding.
        image_shape: [H, W, C] after resizing and padding
        window: (y1, x1, y2, x2) in pixels. The area of the image where the real
                image is (excluding the padding)
        scale: The scaling factor applied to the original image (float32)
        active_class_ids: List of class_ids available in the dataset from which
            the image came. Useful if training on images from multiple datasets
            where not all classes are present in all datasets.
        """
        meta = np.array(
            [image_id] +  # size=1
            list(original_image_shape) +  # size=3
            list(image_shape) +  # size=3
            list(window) +  # size=4 (y1, x1, y2, x2) in image cooredinates
            [scale] +  # size=1
            list(active_class_ids)  # size=num_classes
        )
        return meta

    def parse_image_meta(meta):
        """Parses an array that contains image attributes to its components.
        See compose_image_meta() for more details.

        meta: [batch, meta length] where meta length depends on NUM_CLASSES

        Returns a dict of the parsed values.
        """
        image_id = meta[:, 0]
        original_image_shape = meta[:, 1:4]
        image_shape = meta[:, 4:7]
        window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
        scale = meta[:, 11]
        active_class_ids = meta[:, 12:]
        return {
            "image_id": image_id.astype(np.int32),
            "original_image_shape": original_image_shape.astype(np.int32),
            "image_shape": image_shape.astype(np.int32),
            "window": window.astype(np.int32),
            "scale": scale.astype(np.float32),
            "active_class_ids": active_class_ids.astype(np.int32),
        }

    def parse_image_meta_graph(meta):
        """Parses a tensor that contains image attributes to its components.
        See compose_image_meta() for more details.

        meta: [batch, meta length] where meta length depends on NUM_CLASSES

        Returns a dict of the parsed tensors.
        """
        image_id = meta[:, 0]
        original_image_shape = meta[:, 1:4]
        image_shape = meta[:, 4:7]
        window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
        scale = meta[:, 11]
        active_class_ids = meta[:, 12:]
        return {
            "image_id": image_id,
            "original_image_shape": original_image_shape,
            "image_shape": image_shape,
            "window": window,
            "scale": scale,
            "active_class_ids": active_class_ids,
        }

    def mold_image(images, config):
        """Expects an RGB image (or array of images) and subtracts
        the mean pixel and converts it to float. Expects image
        colors in RGB order.
        """
        return images.astype(np.float32) - config.MEAN_PIXEL

    def trim_zeros_graph(boxes, name='trim_zeros'):
        """Often boxes are represented with matrices of shape [N, 4] and
        are padded with zeros. This removes zero boxes.

        boxes: [N, 4] matrix of boxes.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep
        """
        non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
        boxes = tf.boolean_mask(boxes, non_zeros, name=name)
        return boxes, non_zeros

    def norm_boxes_graph(boxes, shape):
        """Converts boxes from pixel coordinates to normalized coordinates.
        boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
        shape: [..., (height, width)] in pixels

        Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
        coordinates it's inside the box.

        Returns:
            [..., (y1, x1, y2, x2)] in normalized coordinates
        """
        h, w = tf.split(tf.cast(shape, tf.float32), 2)
        scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
        shift = tf.constant([0., 0., 1., 1.])
        return tf.divide(boxes - shift, scale)

    def detection(DEVICE, config, weights_path, image, image_meta):
        """
        this perform segmentation using the trained model

        Args:
            DEVICE[string]: defines the GPU/CPU id
            config[config object]: configurations used for inferrence
            weights_path[string]: full path for the pre-trained weights for loading
            image[ndarray]: numpy array of test image
            image_meta[list]: image meta data for the input image, including image size, resize information
                        the input image was resized for model performing detection

        Returns:
            results[list of dictionary]: segmentation results including 1) central point coordinates for each instance, 2) instance class, 3) confidences for each
                     instance, 4) segmentation mask for each instance
        """
        with tf.device(DEVICE):  # construct MaskRCNN model
            model = MaskRCNN(mode='inference',
                             config=config)  # setting model to inferencing mode and the configuration

        # load trained model for cell segmentation
        print("Loading model ", weights_path)
        model.load_weights(weights_path, by_name=True)
        # Run segmentation
        results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)
        del model  # clear model for next computation
        return results

    def update_prediction_result(results, meta, padding):
        """
        Reformat the segmentation results

        Args:
            results[list]: segmentation result output from the model detection
            meta[list]: image meta data for the input image, including image size, resize information
                        the input image was resized for model performing detection
            padding[list]: padding size for width, height, and the third dimension. the input image was padded to
                           confront the input image size requirement

        Returns:
            r[dict]: segmentation results including 1) instance coordinates, 2) instance class, 3) confidence for each
                     instance, 4) indices for each segmentation contour for both resized image and original size of
                     input image 5) mask size for both sized image and original input image
        """
        scale_used = meta[11]   # scale used to resize the input image for preparing for segmentation
        scale = 1/scale_used
        r = results[0]
        mask_infer = r['masks']  # take segmentation results, which is a stacked masks; each mask is for one instance
        masks_size = mask_infer.shape
        numofmasks = masks_size[2]
        masksIdx = []
        masksIdxOri = []
        for i in range(0, numofmasks):  # convert segmentation masks to indices and save the indices
            masks_indx = np.where(mask_infer[:, :, i] == True)
            masksIdx.append(masks_indx)
        for j in range(0, numofmasks):  # format the indices to match the original input image
            mask_infer_single = mask_infer[:, :, j]
            mask_recon = np.zeros((masks_size[0]-padding[0][0]-padding[0][1], masks_size[1]-padding[1][0]-padding[1][1]), dtype=bool)
            masks_indx = np.where(mask_infer_single==True)
            mask_recon[masks_indx[0] - padding[0][1], masks_indx[1] - padding[1][1]] = True
            mask_int = mask_recon.astype(int)
            mask_int = np.array(mask_int, dtype='uint8')
            w = round(mask_int.shape[0]*scale).astype(int)
            h = round(mask_int.shape[1]*scale).astype(int)
            masks_ori = cv2.resize(mask_int, (h, w))
            masks_indx_ori = np.where(masks_ori == 1)
            mask_ori_size = masks_ori.shape
            masksIdxOri.append(masks_indx_ori)
        r.update({'masksindex': masksIdx})
        r.update({'masksindexori': masksIdxOri})
        r.update({'maskssize': masks_size})
        r.update({'masksorisize': mask_ori_size})
        del r['masks']
        return r

    ############################################################
    #  main body
    ############################################################
    # make result path
    if os.path.isdir(RESULT_PATH) == False:
        os.mkdir(RESULT_PATH)

    # test image path
    TESTIMAGE_DIR = 'TESTIMG'
    COLOR_DIR = 'COLORIMG'
    TESTIMAGE_PATH_M = os.path.join(RESULT_PATH, TESTIMAGE_DIR)
    if os.path.isdir(TESTIMAGE_PATH_M) == False:
        os.mkdir(TESTIMAGE_PATH_M)
    TESTIMAGE_PATH = os.path.join(TESTIMAGE_PATH_M, COLOR_DIR)
    if os.path.isdir(TESTIMAGE_PATH) == False:
        os.mkdir(TESTIMAGE_PATH)

    # write synthesized color image to the test image path
    split_name = DAPI_filename.split('.', 1)
    writename = split_name[0]
    colorwritename = writename + '_COLOR.png'
    colorimage_writename = os.path.join(TESTIMAGE_PATH, colorwritename)
    cv2.imwrite(colorimage_writename, colorimage)

    # Inference Configuration
    config = InferenceConfig()

    # defining the output filenames
    infer_filename = RESULT_CELL
    infer_writefilename = os.path.join(RESULT_PATH, infer_filename)
    infer2_filename = RESULT_NUCLEI
    infer2_writefilename = os.path.join(RESULT_PATH, infer2_filename)
    image_name = 'image.dat'
    image_bio_name = 'imagebio.dat'
    image_writename = os.path.join(RESULT_PATH, image_name)
    image_bio_writename = os.path.join(RESULT_PATH, image_bio_name)
    # formating output filenames to the filenames [dictionary], which will be sent as response to the plug-in for retrieving the results
    filenames = {'image': image_writename,
                 'infer': infer_writefilename,
                 'infer2': infer2_writefilename,
                 'imagebio': image_bio_writename,
                 'biomarkerpath': BIOMARKER_path,
                 'RESULT_PATH': RESULT_PATH}

    # if the cell and nucleus segmentation file are existed and 'SEGMENTATION' is set to 'NO', load the segmentation results
    # otherwise, perform segmentation.
    if os.path.isfile(infer_writefilename) and os.path.isfile(infer2_writefilename) and SEGMENTATION == 'NO':
        print('using existing segmentation results')

    # perform segmentation
    else:
        ### --- step one: cell segmentation --- ###
        # load the test image, which is the synthesized color image just written
        dataset = Dataset()   # construct dataset object
        dataset.load_data(TESTIMAGE_PATH, "val", 'cell')  # specify the test image
        dataset.prepare()   # for detection
        image_id = dataset.image_ids[0]
        image, image_meta, padding = load_image_only(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]
        print("image ID: {}".format(info["id"]))
        print("Original image shape: ", parse_image_meta(image_meta[np.newaxis, ...])["original_image_shape"][0])

        # trained model path for cell segmentation
        weights_path = os.path.join(MODEL_PATH, 'mask_rcnn_cell.h5')

        # run cell segmentation
        try:  # perform segmentation using GPU, if cannot use GPU run properly, use CPU
            DEVICE = "/device:gpu:0"
            results = detection(DEVICE, config, weights_path, image, image_meta)
        except:
            DEVICE = '/device:CPU:0'
            results = detection(DEVICE, config, weights_path, image, image_meta)


        # format and save the results
        r = update_prediction_result(results, image_meta, padding)
        with open(infer_writefilename, 'wb') as f:
            pickle.dump(r, f, protocol=2)
        print('finish cell segmentation')

        # save synthesized color image in .dat format
        with open(image_writename, 'wb') as x:
            pickle.dump(colorimage, x, protocol=2)
        # save the biomarker image in .dat format
        with open(image_bio_writename, 'wb') as x:
            pickle.dump(image_BIOMARKER, x, protocol=2)

        ### --- step two: nucleus segmentation --- ###
        # prepare input image for segmentation
        molded_image_resize_nucleus = resize_image(image_dapi,
                    min_scale=config.IMAGE_MIN_SCALE,
                    mode=config.IMAGE_RESIZE_MODE)  # resize the input image to confront the size requirement
        image_test = molded_image_resize_nucleus[0]

        # trained model path for nucleus segmentation
        nucleus_weights_path = os.path.join(MODEL_PATH, "mask_rcnn_nucleus.h5")

        # run nucleus segmentation
        try:
            DEVICE = "/device:gpu:0"
            results_nucleus = detection(DEVICE, config, nucleus_weights_path, image_test, image_meta)
        except:
            DEVICE = '/device:CPU:0'
            results_nucleus = detection(DEVICE, config, nucleus_weights_path, image_test, image_meta)
        print('finish nuclei segmentation')

        # format and save nucleus segmentation results
        r2 = update_prediction_result(results_nucleus, image_meta, padding)
        with open(infer2_writefilename, 'wb') as f:
            pickle.dump(r2, f, protocol=2)

    return json.dumps(filenames) # return the filenames in json format as response; plug-in uses filenames to retrieve the results.

# execute the application
if __name__ == "__main__":
    app.run()



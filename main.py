import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # @param ["tensorflow", "jax", "torch"]

import math
from tensorflow import data as tf_data
import keras
import keras_cv
import numpy as np
from keras_cv import bounding_box
import glob
import xml.etree.ElementTree as ET
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BATCH_SIZE = 1  # Set your batch size
class_mapping = {'6244914': 0}  # Define your class mapping

def display_image_with_boxes(image, boxes):
    image = tf.squeeze(image, axis=0)  # Remove batch dimension
    image = tf.cast(image, tf.float32) / 255.0
    if tf.rank(boxes) != 2 or tf.shape(boxes)[1] != 4:
        print("No bounding boxes to display.")
        plt.imshow(image)
        plt.show()
        return

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    image_height, image_width, _ = image.shape
    for box in boxes:
        # Convert bounding box coordinates to pixel values
        xmin, ymin, xmax, ymax = box
        xmin *= image_width
        xmax *= image_width
        ymin *= image_height
        ymax *= image_height
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
    
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    labels = []

    for member in root.findall('object'):
        label = member.find('name').text
        labels.append(class_mapping[label])  # Convert label to integer
        bbox = member.find('bndbox')
        bbox = [int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text), int(bbox.find('ymax').text)]
        boxes.append(bbox)

    return {'path': root.find('filename').text, 'object': {'bndbox': boxes, 'name': labels}}

def load_image(image_path):
    return np.array(Image.open(image_path))

def load_dataset(data_dir):
    xml_files = glob.glob(os.path.join(data_dir, 'Annotations', '*.xml'))
    data = [parse_xml(xml_file) for xml_file in xml_files]

    images = []
    bbox = []
    labels = []

    for item in data:
        image_path = os.path.join(data_dir, 'train', item['path'])
        image = load_image(image_path)
        print(f"Loaded image shape: {image.shape}")  # Print the shape of the loaded image
        images.append(image)
        bbox.append(np.array(item['object']['bndbox'], dtype=np.int32))  # Ensure bounding boxes are integers
        labels.append(np.array(item['object']['name'], dtype=np.int32))  # Ensure labels are integers

    images = np.stack(images, axis=0)  # Convert list of images to 4D numpy array
    ds = tf.data.Dataset.from_tensor_slices((images, {'boxes': bbox, 'classes': labels}))
    return ds

data_dir = 'data/6244914'  # Replace with the path to your data directory
train_ds = load_dataset(data_dir)
eval_ds = load_dataset(data_dir)

# Shuffle, batch, and prefetch the dataset
#train_ds = train_ds.shuffle(BATCH_SIZE * 4)

# Define augmentation layers
augmenters = [
    keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
    keras_cv.layers.RandomRotation(factor=0.15),  # Rotate the image by +/- 15 degrees
]

def create_augmenter_fn(augmenters):
    def augmenter_fn(image, label):
        for augmenter in augmenters:
            image = augmenter(image)
        return image, label

    return augmenter_fn

augmenter_fn = create_augmenter_fn(augmenters)
train_ds = train_ds.map(augmenter_fn, num_parallel_calls=tf_data.AUTOTUNE)

# Define the dict_to_tuple function
def dict_to_tuple(images, bounding_boxes):
    if bounding_boxes is None:
        print("Warning: bounding_boxes is None")
        dense_bounding_boxes = None
    else:
        try:
            bounding_boxes_dict = {'boxes': bounding_boxes['boxes'], 'classes': bounding_boxes['classes']}
            dense_bounding_boxes = bounding_box.to_dense(bounding_boxes_dict, max_boxes=32)
        except KeyError as e:
            print(f"KeyError: {e}")
            print(f"bounding_boxes: {bounding_boxes}")  # Print the bounding_boxes dictionary
            dense_bounding_boxes = None
    return images, dense_bounding_boxes

# No need to resize images in the dataset
train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

# Prefetch the data for better performance
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
eval_ds = eval_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)



base_lr = 0.005
# including a global_clipnorm is extremely important in object detection tasks
optimizer = keras.optimizers.SGD(
    learning_rate=base_lr, momentum=0.9, global_clipnorm=10.0
)

model = keras_cv.models.YOLOV8Detector.from_preset(
    "resnet50_imagenet",
    bounding_box_format="xyxy",
    num_classes=1,
)

model.compile(
    classification_loss="binary_crossentropy",
    box_loss="ciou",
    optimizer=optimizer,
)
print(next(iter(train_ds)))
print(next(iter(eval_ds)))


print(math.ceil(len(train_ds) / BATCH_SIZE))
print(math.ceil(len(eval_ds) / BATCH_SIZE))
model.fit(
    train_ds,
    epochs=10,
    steps_per_epoch=math.ceil(len(train_ds) / BATCH_SIZE)
)

# Save the full model to a new .h5 file
model.save('full_yolomodel', save_format='tf')
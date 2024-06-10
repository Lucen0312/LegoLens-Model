import os
import glob
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util

IMAGE_DIR = '/data/images'
ANNOTATIONS_DIR = '/data/annotations'

LABEL_MAP = {
    '6244914': 6244914,
    # Add more classes as needed
}

def class_text_to_int(row_label):
    return LABEL_MAP.get(row_label, None)

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(os.path.join(path, '**', '*.xml'), recursive=True):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (os.path.join(IMAGE_DIR, root.find('filename').text),
                     int(root.find('size/width').text),
                     int(root.find('size/height').text),
                     member.find('name').text,
                     int(member.find('bndbox/xmin').text),
                     int(member.find('bndbox/ymin').text),
                     int(member.find('bndbox/xmax').text),
                     int(member.find('bndbox/ymax').text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def create_tf_example(group, path):
    with tf.io.gfile.GFile(group.filename, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = tf.io.BytesIO(encoded_jpg)
    image = tf.io.decode_jpeg(encoded_jpg_io.getvalue())

    width = int(group.width)
    height = int(group.height)

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    writer = tf.io.TFRecordWriter('train.record')
    examples = xml_to_csv(ANNOTATIONS_DIR)
    grouped = examples.groupby('filename')
    
    for filename, group in grouped:
        tf_example = create_tf_example(group, IMAGE_DIR)  
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecord file.')

if __name__ == '__main__':
    tf.compat.v1.app.run()
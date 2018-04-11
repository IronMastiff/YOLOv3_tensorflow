from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np

def xml_extractor( dir ):
    DOMTree = parse( dir )
    collection = DOMTree.documentElement
    file_name_xml = collection.getElementsByTagName( 'filename' )[0]
    objects_xml = collection.getElementsByTagName( 'object' )
    size_xml = collection.getElementsByTagName( 'size' )

    file_name = file_name_xml.childNodes[0].data

    for size in size_xml:
        width = size.getElementsByTagName( 'width' )[0]
        height = size.getElementsByTagName( 'height' )[0]

        width = width.childNodes[0].data
        height = height.childNodes[0].data

    objects = []
    for object_xml in objects_xml:
        object_name = object_xml.getElementsByTagName( 'name' )[0]
        bdbox = object_xml.getElementsByTagName( 'bndbox' )[0]
        xmin = bdbox.getElementsByTagName( 'xmin' )[0]
        ymin = bdbox.getElementsByTagName( 'ymin' )[0]
        xmax = bdbox.getElementsByTagName( 'xmax' )[0]
        ymax = bdbox.getElementsByTagName( 'ymax' )[0]

        object = ( object_name.childNodes[0].data,
                   xmin.childNodes[0].data,
                   ymin.childNodes[0].data,
                   xmax.childNodes[0].data,
                   ymax.childNodes[0].data )

        objects.append( object )

    return file_name, width, height, objects

def labels_normaliszer( batches_filenames, target_width, target_height ):

    class_map = {
        'person' : 4,
        'bird' : 5,
        'cat' : 6,
        'cow' : 7,
        'dog' : 8,
        'horse' : 9,
        'sheep' : 10,
        'aeroplane' : 11,
        'bicycle' : 12,
        'boat' : 13,
        'bus' : 14,
        'car' : 15,
        'motorbike' : 16,
        'train' : 17,
        'bottle' : 18,
        'chair' : 19,
        'dining table' : 20,
        'potted plant': 21,
        'sofa' : 22,
        'tv/monitor' : 23
    }

    batches_labels = []
    batches_height_width = []
    batch_labels = []
    batch_height_width = []
    labels = []
    height_width = []
    for batch_filenames in batches_filenames:
        for filename in batch_filenames:
            _, width, height, objects = xml_extractor( filename )
            width_proprotion = 1.0 * target_width / width
            height_proprotion = 1.0 * target_height / height

            label = np.zeros( [target_height, target_width, 255] )
            for object in objects:
                class_label = class_map[object[0]]
                xmin = object[1]
                ymin = object[2]
                xmax = object[3]
                ymax = object[4]

                x = ( 1.0 * xmax + xmin ) / 2 * width_proprotion
                y = ( 1.0 * ymax + ymin ) / 2 * height_proprotion

                bdbox_width = ( 1.0 * xmax - xmin ) * width_proprotion
                bdbox_height = ( 1.0 * ymax - ymin ) * height_proprotion

                label[int( y )][int( x )][0] = x    # Point x
                label[int( y )][int( x )][1] = y    # Point y
                label[int( y )][int( x )][2] = bdbox_width    # bdbox width
                label[int( y )][int( x )][3] = bdbox_height    # bdbox_height
                label[int( y )][int( x )][4] = 0.5    # objectness
                label[int( y )][int( x )][class_label] = 0.9    # class label

                labels.append( label )
            batch_labels.append( labels )
        batches_labels.append( batch_labels )

        batches_labels = np.array( batches_labels )

    return batches_labels






'''--------Test extract_labels--------'''
if __name__ == '__main__':
    file_name, width, height, objects = xml_extractor( '../data/VOCtest_06-Nov-2007/Annotations/000001.xml' )
    print( file_name, '\n', width, '\n', height, '\n', objects )
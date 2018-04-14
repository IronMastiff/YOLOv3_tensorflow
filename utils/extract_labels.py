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

def labels_normalizer( batches_filenames, target_width, target_height, layerout_width, layerout_height ):

    class_map = {
        'person' : 5,
        'bird' : 6,
        'cat' : 7,
        'cow' : 8,
        'dog' : 9,
        'horse' : 10,
        'sheep' : 11,
        'aeroplane' : 12,
        'bicycle' : 13,
        'boat' : 14,
        'bus' : 15,
        'car' : 16,
        'motorbike' : 17,
        'train' : 18,
        'bottle' : 19,
        'chair' : 20,
        'diningtable' : 21,
        'pottedplant': 22,
        'sofa' : 23,
        'tvmonitor' : 24
    }

    height_width = []
    batches_labels = []
    for batch_filenames in batches_filenames:
        batch_labels = []
        for filename in batch_filenames:
            _, width, height, objects = xml_extractor( filename )
            width_preprotion = target_width / int( width )
            height_preprotion = target_height / int( height )
            label = np.add( np.zeros( [int( layerout_height ), int( layerout_width ), 255] ), 1e-8 )
            for object in objects:
                class_label = class_map[object[0]]
                xmin = float( object[1] )
                ymin = float( object[2] )
                xmax = float( object[3] )
                ymax = float( object[4] )
                x = ( 1.0 * xmax + xmin ) / 2 * width_preprotion
                y = ( 1.0 * ymax + ymin ) / 2 * height_preprotion
                bdbox_width = ( 1.0 * xmax - xmin ) * width_preprotion
                bdbox_height = ( 1.0 * ymax - ymin ) * height_preprotion
                falg_width = int( target_width ) / layerout_width
                flag_height = int( target_height ) / layerout_height
                box_x = x // falg_width
                box_y = y // flag_height
                if box_x == layerout_width:
                    box_x -= 1
                if box_y == layerout_height:
                    box_y -= 1
                for i in range( 3 ):
                    label[int( box_y ), int( box_x ), i * 25] = x    # point x
                    label[int( box_y ), int( box_x ), i * 25 + 1] = y    # point y
                    label[int( box_y ), int( box_x ), i * 25 + 2] = bdbox_width    # bdbox width
                    label[int( box_y ), int( box_x ), i * 25 + 3] = bdbox_height    # bdbox height
                    label[int( box_y ), int( box_x ), i * 25 + 4] = 1    # objectness
                    label[int( box_y ), int( box_x ), i * 25 + int( class_label )] = 0.9    # class label

            batch_labels.append( label )

        batches_labels.append( batch_labels )

    # batches_labels = np.array( batches_labels )

    return batches_labels






'''--------Test extract_labels--------'''
if __name__ == '__main__':
    dir = [['../data/VOCtest_06-Nov-2007/Annotations/000001.xml', '../data/VOCtest_06-Nov-2007/Annotations/000002.xml'], ['../data/VOCtest_06-Nov-2007/Annotations/000003.xml', '../data/VOCtest_06-Nov-2007/Annotations/000004.xml']]
    batches_labels = labels_normalizer( dir, 512, 512, 16, 16 )
    print( np.array( dir ).shape )
    print( np.array( batches_labels ).shape )
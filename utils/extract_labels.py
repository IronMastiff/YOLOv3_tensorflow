from xml.dom.minidom import parse
import xml.dom.minidom

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


'''--------Test extract_labels--------'''
if __name__ == '__main__':
    file_name, width, height, objects = xml_extractor( '../data/VOCtest_06-Nov-2007/Annotations/000001.xml' )
    print( file_name, '\n', width, '\n', height, '\n', objects )
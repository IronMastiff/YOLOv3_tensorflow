import numpy as np

def calculate_min( point, data ):
    min_point = point - data

    return min_point

def calculate_max( point, data ):
    max_point = point + data

    return max_point

def IOU_calculator( x, y, width, height, l_x, l_y, l_width, l_height ):
    '''
    Cculate IOU

    :param x: net predicted x
    :param y: net predicted y
    :param width: net predicted width
    :param height: net predicted height
    :param l_x: label x
    :param l_y: label y
    :param l_width: label width
    :param l_height: label height
    :return: IOU
    '''

    x_max = calculate_max( x , width / 2 )
    y_max = calculate_max( y, height / 2 )
    x_min = calculate_min( x, width / 2 )
    y_min = calculate_min( y, height / 2 )

    l_x_max = calculate_max( l_x, width / 2 )
    l_y_max = calculate_max( l_y, height / 2 )
    l_x_min = calculate_min( l_x, width / 2 )
    l_y_min = calculate_min( l_y, height / 2 )

    '''--------Caculate Both Area's point--------'''
    xend = min( x_max, l_x_max )
    xstart = max( x_min, l_x_min )

    yend = min( y_max, l_y_max )
    ystart = max( y_min, l_y_min )

    area_width = xend - xstart
    area_height = yend - ystart

    '''--------Caculate the IOU--------'''
    if area_width < 0 or area_height < 0:
        IOU = 0
    else:
        area = area_width * area_height

        IOU = area / ( width * height + l_width * l_height - area )

    return IOU



'''--------Test the IOU function--------'''
if __name__ == '__main__':
    IOU = IOU_calculator( 1, 1, 2, 2, 2, 2, 2, 2 )
    print( IOU )
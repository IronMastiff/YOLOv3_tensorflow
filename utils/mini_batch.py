import tensorflow as tf
import numpy as np

def mini_batch( batch_size, inputs ):
    batches = []
    num = inputs.shape[0] // batch_size

    for i in range( num ):
        batch = inputs[i * batch_size : ( i + 1 ) * batch_size, :, :, : ]

        batches.append( batch )

    if inputs.shape[0] % batch_size:
        batch = inputs[( i + 1 ) * batch_size :, :, :, :]

        batches.append( batch )

    return batches


'''--------test Minibatch function--------'''
if __name__ == '__main__':
    array = np.random.random( ( 3, 1, 1, 1 ) )
    batches = mini_batch( 1, array )
    print( batches )
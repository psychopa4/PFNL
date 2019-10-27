
import tensorflow as tf

def weight_from_caffe(caffenet):
    def func(shape, dtype):
        sc = tf.get_variable_scope()
        name = sc.name.split('/')[-1]
        print ('init: ', name, shape, caffenet.params[name][0].data.shape)
        return tf.transpose(caffenet.params[name][0].data, perm=[2 ,3 ,1 ,0])
    return func

def bias_from_caffe(caffenet):
    def func(shape, dtype):
        sc = tf.get_variable_scope()
        name = sc.name.split('/')[-1]
        return caffenet.params[name][1].data
    return func

def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.2))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)
        
def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))



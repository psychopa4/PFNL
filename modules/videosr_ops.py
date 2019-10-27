import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim


def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)


def get_shape(x):
    shape = tf.shape(x)
    check = tf.Assert(tf.reduce_all(shape >= 0), ["EASYFLOW: Need value.shape >= 0, got ", shape])
    shape = control_flow_ops.with_dependencies([check], shape)
    return [shape[i] for i in range(shape.shape.as_list()[0])]


def zero_upsampling(x, scale_factor):
    dims = x.get_shape().as_list()
    if len(dims) == 5:
        n, t, h, w, c = dims
        y = tf.concat([x] + [tf.zeros_like(x)] * (scale_factor ** 2 - 1), -1)
        y = tf.reshape(y, [n, t, h, w, scale_factor, scale_factor, c])
        y = tf.transpose(y, [0, 1, 2, 4, 3, 5, 6])
        y = tf.reshape(y, [n, t, h * scale_factor, w * scale_factor, c])
    elif len(dims) == 4:
        n, h, w, c = dims
        y = tf.concat([x] + [tf.zeros_like(x)] * (scale_factor ** 2 - 1), -1)
        y = tf.reshape(y, [n, h, w, scale_factor, scale_factor, c])
        y = tf.transpose(y, [0, 1, 3, 2, 4, 5])
        y = tf.reshape(y, [n, h * scale_factor, w * scale_factor, c])
    return y


def leaky_relu(x, alpha=0.1):
    return tf.maximum(x, alpha * x)


def prelu(x):
    alphas = tf.get_variable('alpha', x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - tf.abs(x)) * 0.5

    return pos + neg


def display_tf_variables(train_vars):
    print('Training Variables: ')
    for var in train_vars:
        print('\t', var.name)


def resize_images(images, size, method=2, align_corners=False):
    dims = len(images.get_shape())
    if dims == 5:
        n, t, h, w, c = images.get_shape().as_list()
        images = tf.reshape(images, [n * t, h, w, c])
    images = tf.image.resize_images(images, size, method, align_corners)
    if dims == 5:
        images = tf.reshape(images, [n, t, size[0], size[1], c])
    return images


def rgb2y(inputs):
    with tf.name_scope('rgb2y'):
        if inputs.get_shape()[-1].value == 1:
            return inputs
        assert inputs.get_shape()[-1].value == 3, 'Error: rgb2y input should be RGB or grayscale!'
        dims = len(inputs.get_shape())
        if dims == 4:
            scale = tf.reshape([65.481, 128.553, 24.966], [1, 1, 1, 3]) / 255.0
        elif dims == 5:
            scale = tf.reshape([65.481, 128.553, 24.966], [1, 1, 1, 1, 3]) / 255.0
        output = tf.reduce_sum(inputs * scale, reduction_indices=dims - 1, keep_dims=True)
        output = output + 16 / 255.0
    return output


def rgb2ycbcr(inputs):
    with tf.name_scope('rgb2ycbcr'):
        if inputs.get_shape()[-1].value == 1:
            return inputs
        assert inputs.get_shape()[-1].value == 3, 'Error: rgb2ycbcr input should be RGB or grayscale!'
        ndims = len(inputs.get_shape())
        origT = [[65.481, 128.553, 24.966], [-37.797, -74.203, 112], [112, -93.786, -18.214]]
        origOffset = [16.0, 128.0, 128.0]
        if ndims == 4:
            origT = [tf.reshape(origT[i], [1, 1, 1, 3]) / 255.0 for i in range(3)]
        elif ndims == 5:
            origT = [tf.reshape(origT[i], [1, 1, 1, 1, 3]) / 255.0 for i in range(3)]
        output = []
        for i in range(3):
            output.append(tf.reduce_sum(inputs * origT[i], reduction_indices=-1, keep_dims=True) + origOffset[i] / 255.0)
        return tf.concat(output, -1)


def ycbcr2rgb(inputs):
    with tf.name_scope('ycbcr2rgb'):
        if inputs.get_shape()[-1].value == 1:
            return inputs
        assert inputs.get_shape()[-1].value == 3, 'Error: rgb2ycbcr input should be RGB or grayscale!'
        ndims = len(inputs.get_shape())
        # origT = np.array([[65.481, 128.553, 24.966], [-37.797 -74.203 112], [112 -93.786 -18.214]])
        # T = tf.inv(origT)
        Tinv = [[0.00456621, 0., 0.00625893], [0.00456621, -0.00153632, -0.00318811], [0.00456621, 0.00791071, 0.]]
        origOffset = [16.0, 128.0, 128.0]
        if ndims == 4:
            origT = [tf.reshape(Tinv[i], [1, 1, 1, 3]) * 255.0 for i in range(3)]
            origOffset = tf.reshape(origOffset, [1, 1, 1, 3]) / 255.0
        elif ndims == 5:
            origT = [tf.reshape(Tinv[i], [1, 1, 1, 1, 3]) * 255.0 for i in range(3)]
            origOffset = tf.reshape(origOffset, [1, 1, 1, 1, 3]) / 255.0
        output = []
        for i in range(3):
            output.append(tf.reduce_sum((inputs - origOffset) * origT[i], reduction_indices=-1, keep_dims=True))
        return tf.concat(output, -1)
    

def rgb2gray(inputs):
    with tf.name_scope('rgb2gray'):
        if inputs.get_shape()[-1].value == 1:
            return inputs
        assert inputs.get_shape()[-1].value == 3, 'Error: rgb2y input should be RGB or grayscale!'
        dims = len(inputs.get_shape())
        if dims == 4:
            scale = tf.reshape([0.299, 0.587, 0.114], [1, 1, 1, 3])
        elif dims == 5:
            scale = tf.reshape([0.299, 0.587, 0.114], [1, 1, 1, 1, 3])
        output = tf.reduce_sum(inputs * scale, reduction_indices=dims - 1, keep_dims=True)
    return output


def flowToColor(flow, maxflow=None):
    def makeColorwheel():
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR

        colorwheel = np.zeros([ncols, 3], dtype=np.float32)  # r g b

        col = 0
        # RY
        colorwheel[0:RY, 0] = 255.0
        colorwheel[0:RY, 1] = np.floor(np.multiply(255.0 / RY, range(RY)))
        col = col + RY
        # YG
        colorwheel[col + np.arange(0, YG), 0] = 255.0 - np.floor(np.multiply(255.0 / YG, range(YG)))
        colorwheel[col + np.arange(0, YG), 1] = 255.0
        col = col + YG
        # GC
        colorwheel[col + np.arange(0, GC), 1] = 255.0
        colorwheel[col + np.arange(0, GC), 2] = np.floor(np.multiply(255.0 / GC, range(GC)))
        col = col + GC
        # CB
        colorwheel[col + np.arange(0, CB), 1] = 255.0 - np.floor(np.multiply(255.0 / CB, range(CB)))
        colorwheel[col + np.arange(0, CB), 2] = 255.0
        col = col + CB
        # BM
        colorwheel[col + np.arange(0, BM), 2] = 255.0
        colorwheel[col + np.arange(0, BM), 0] = np.floor(np.multiply(255.0 / BM, range(BM)))
        col = col + BM
        # MR
        colorwheel[col + np.arange(0, MR), 2] = 255.0 - np.floor(np.multiply(255.0 / MR, range(MR)))
        colorwheel[col + np.arange(0, MR), 0] = 255.0
        return colorwheel

    def atan2(y, x):
        angle = tf.where(tf.greater(x, 0.0), tf.atan(y / x), tf.zeros_like(x))
        angle = tf.where(tf.logical_and(tf.less(x, 0.0), tf.greater_equal(y, 0.0)), tf.atan(y / x) + np.pi, angle)
        angle = tf.where(tf.logical_and(tf.less(x, 0.0), tf.less(y, 0.0)), tf.atan(y / x) - np.pi, angle)
        angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.greater(y, 0.0)), 0.5 * np.pi * tf.ones_like(x), angle)
        angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.less(y, 0.0)), -0.5 * np.pi * tf.ones_like(x), angle)
        angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.equal(y, 0.0)), np.nan * tf.zeros_like(x), angle)
        return angle
    eps = 2.2204e-16

    u = flow[:, :, :, 0]
    v = flow[:, :, :, 1]

    if maxflow is not None:
        maxrad = maxflow
    else:
        rad = tf.sqrt(u ** 2 + v ** 2)
        maxrad = tf.reduce_max(rad)

    u /= (maxrad + eps)
    v /= (maxrad + eps)
    rad = tf.sqrt(u ** 2 + v ** 2)
    
    colorwheel = makeColorwheel()
    ncols = colorwheel.shape[0]

    a = atan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)  # -1~1 maped to 0 ~ ncols-1
    k0 = tf.floor(fk)
    k1 = (k0 + 1) % ncols
    # k1[k1 == ncols] = 0
    f = fk - k0
    k0 = tf.cast(k0, tf.int32)
    k1 = tf.cast(k1, tf.int32)

    
    col0 = tf.gather(colorwheel, k0) / 255.0
    col1 = tf.gather(colorwheel, k1) / 255.0
    f = tf.expand_dims(f, dim=-1)
    col = (1 - f) * col0 + f * col1

    idx = tf.tile(tf.expand_dims(rad <= 1, dim=-1), [1, 1, 1, 3])
    rad = tf.expand_dims(rad, dim=-1)
    col = tf.where(idx, 1 - rad * (1 - col), col * 0.75)

    img = tf.cast(tf.floor(255.0 * col), tf.uint8)
    return img


def channel2sub(x, scale_factor):
    dims = len(x.get_shape())
    if dims == 5:
        num_batch, num_frame, height, width, num_channels = map(lambda x: x.value, x.get_shape())
        out_height = height * scale_factor
        out_width = width * scale_factor
        out_channels = num_channels / scale_factor / scale_factor
        x = tf.reshape(x, [num_batch, num_frame, height, width, scale_factor, scale_factor, out_channels])
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3, 5, 6])
        x = tf.reshape(x, [num_batch, num_frame, out_height, out_width, out_channels])
    else:
        num_batch, height, width, num_channels = map(lambda x: x.value, x.get_shape())
        out_height = height * scale_factor
        out_width = width * scale_factor
        out_channels = num_channels / scale_factor / scale_factor
        x = tf.reshape(x, [num_batch, height, width, scale_factor, scale_factor, out_channels])
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [num_batch, out_height, out_width, out_channels])
    return x


def sub2channel(x, scale_factor):
    dims = len(x.get_shape())
    if dims == 5:
        num_batch, num_frame, out_height, out_width, num_channels = map(lambda x: x.value, x.get_shape())
        height = out_height / scale_factor
        width = out_width / scale_factor
        x = tf.reshape(x, [num_batch, num_frame, height, scale_factor, width, scale_factor, num_channels])
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3, 5, 6])
        x = tf.reshape(x, [num_batch, num_frame, height, width, scale_factor * scale_factor * num_channels])
    else:
        num_batch, out_height, out_width, num_channels = map(lambda x: x.value, x.get_shape())
        height = out_height / scale_factor
        width = out_width / scale_factor
        x = tf.reshape(x, [num_batch, height, scale_factor, width, scale_factor, num_channels])
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [num_batch, height, width, scale_factor * scale_factor * num_channels])
    return x


def _repeat(x, n_repeats):
    with tf.variable_scope('_repeat'):
        # rep = tf.transpose(
        #     tf.expand_dims(tf.ones(shape=tf.pack([n_repeats, ])), 1), [1, 0])
        # rep = tf.cast(rep, 'int32')
        # with tf.device('/cpu:0'):
        #     x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        # return tf.reshape(x, [-1])
        x = tf.reshape(x, [-1, 1])
        with tf.device('/cpu:0'):
            res = tf.tile(x, [1, n_repeats])
        res = tf.reshape(res, [-1])
        res = tf.cast(res, 'int32')
    return res


def meshgrid(height, width):
    with tf.variable_scope('_meshgrid'):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

        # with tf.device('/cpu:0'):
        #     x_t = tf.matmul(tf.ones(shape=tf.pack([height, 1])),
        #                     tf.transpose(tf.expand_dims(tf.linspace(0.0, -1.0 + width, width), 1), [1, 0]))
        #     y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0, -1.0 + height, height), 1),
        #                     tf.ones(shape=tf.pack([1, width])))
        # x_t = tf.expand_dims(x_t, 2)
        # y_t = tf.expand_dims(y_t, 2)
        # grid = tf.concat(2, [x_t, y_t])
        with tf.device('/cpu:0'):
            grid = tf.meshgrid(list(range(height)), list(range(width)), indexing='ij')
            grid = tf.cast(tf.stack(grid, axis=2)[:, :, ::-1], tf.float32)
    return grid


def imwarp_backward(uv, input_dim, out_size):
    def _interpolate_backward(im, x, y, out_size):
        with tf.variable_scope('_interp_b', reuse=False):
            # constants
            num_batch, height, width, channels = map(lambda x: x.value, im.get_shape())
            out_height = out_size[0]
            out_width = out_size[1]

            x = tf.cast(x, 'float32') * (out_height / height)
            y = tf.cast(y, 'float32') * (out_width / width)
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(out_height - 1, 'int32')
            max_x = tf.cast(out_width - 1, 'int32')

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = out_width
            dim1 = out_width * out_height

            base = _repeat(tf.range(num_batch) * dim1, height * width)
            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
            wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
            wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
            wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
            output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
            return output

    with tf.variable_scope('imwarp_b'):
        dims = len(input_dim.get_shape())
        if dims == 5:
            n, num_frame, height, width, num_channels = input_dim.get_shape().as_list()
            input_dim = tf.reshape(input_dim, [n * num_frame, height, width, num_channels])
        dims_uv = len(uv.get_shape())
        if dims_uv == 5:
            n_uv, num_frame_uv, height_uv, width_uv, num_channels_uv = uv.get_shape().as_list()
            uv = tf.reshape(uv, [n_uv * num_frame_uv, height_uv, width_uv, num_channels_uv])

        num_batch, height, width, num_channels = map(lambda x: x.value, input_dim.get_shape())
        uv = tf.cast(uv, 'float32')

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        out_height = out_size[0]
        out_width = out_size[1]
        grid = meshgrid(height, width)
        grid = tf.expand_dims(grid, 0)
        grid = tf.tile(grid, tf.stack([num_batch, 1, 1, 1]))

        T_g = grid + uv

        x_s = T_g[:, :, :, 0]
        y_s = T_g[:, :, :, 1]
        x_s_flat = tf.reshape(x_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])

        input_transformed = _interpolate_backward(input_dim, x_s_flat, y_s_flat, out_size)
        # output: (n * h * w * c), output_w: (n * h * w * 1)
        input_transformed = tf.clip_by_value(input_transformed, 0.0, 1.0)
        output = tf.reshape(
            input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
        if dims == 5:
            output = tf.reshape(output, [n, num_frame, height, width, num_channels])
    return output


def imwarp_forward(uv, input_dim, out_size):
    def _interpolate_forward(im, x, y, out_size):
        with tf.variable_scope('_interp_f', reuse=False):
            # constants
            num_batch, height, width, channels = map(lambda x: x.value, im.get_shape())
            out_height = out_size[0]
            out_width = out_size[1]

            x = tf.cast(x, 'float32') * (out_height / height)
            y = tf.cast(y, 'float32') * (out_width / width)
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(out_height - 1, 'int32')
            max_x = tf.cast(out_width - 1, 'int32')

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = out_width
            dim1 = out_width * out_height

            base = _repeat(tf.range(num_batch) * dim1, height * width)
            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')

            # and finally calculate interpolated values
            wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
            wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
            wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
            wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)

            # try:
            #     tf.get_variable_scope()._reuse = False
            #     warp_img = tf.get_variable('warp_img', [num_batch * out_height * out_width, channels],
            #                                initializer=tf.constant_initializer(0.0), trainable=False)
            #     tf.get_variable_scope()._reuse = True
            # except ValueError:
            #     tf.get_variable_scope().reuse_variables()
            #     warp_img = tf.get_variable('warp_img', [num_batch * out_height * out_width, channels],
            #                                initializer=tf.constant_initializer(0.0), trainable=False)
            # init0 = tf.group(
            #     tf.assign(warp_img, tf.zeros([num_batch * out_height * out_width, channels], dtype=tf.float32)))
            # with tf.control_dependencies([init0]):
            #     warp_img = tf.scatter_add(warp_img, idx_a, wa * im_flat, name='interp_sa1')
            #     warp_img = tf.scatter_add(warp_img, idx_b, wb * im_flat, name='interp_sa2')
            #     warp_img = tf.scatter_add(warp_img, idx_c, wc * im_flat, name='interp_sa3')
            #     warp_img = tf.scatter_add(warp_img, idx_d, wd * im_flat, name='interp_sa4')

            num_segments = num_batch * out_height * out_width
            with tf.device('/cpu:0'):
                warp_img_a = tf.unsorted_segment_sum(data=wa * im_flat, segment_ids=idx_a, num_segments=num_segments)
                warp_img_b = tf.unsorted_segment_sum(data=wb * im_flat, segment_ids=idx_b, num_segments=num_segments)
                warp_img_c = tf.unsorted_segment_sum(data=wc * im_flat, segment_ids=idx_c, num_segments=num_segments)
                warp_img_d = tf.unsorted_segment_sum(data=wd * im_flat, segment_ids=idx_d, num_segments=num_segments)
            warp_img = warp_img_a + warp_img_b + warp_img_c + warp_img_d
            return warp_img

    with tf.variable_scope('imwarp_f'):
        dims = len(input_dim.get_shape())
        if dims == 5:
            n, num_frame, height, width, num_channels = map(lambda x: x.value, input_dim.get_shape())
            input_dim = tf.reshape(input_dim, [n * num_frame, height, width, num_channels])

        num_batch, height, width, num_channels = map(lambda x: x.value, input_dim.get_shape())
        uv = tf.cast(uv, 'float32')

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        out_height = out_size[0]
        out_width = out_size[1]
        grid = meshgrid(height, width)
        grid = tf.expand_dims(grid, 0)
        grid = tf.tile(grid, tf.stack([num_batch, 1, 1, 1]))

        T_g = grid + uv

        x_s = T_g[:, :, :, 0]
        y_s = T_g[:, :, :, 1]
        x_s_flat = tf.reshape(x_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])

        input_transformed = _interpolate_forward(input_dim, x_s_flat, y_s_flat, out_size)
        # output: n * h * w * c
        output = tf.reshape(input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
        if dims == 5:
            output = tf.reshape(output, [n, num_frame, out_height, out_width, num_channels])
    return output

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    uv = -tf.ones([2, 100, 100, 2], tf.float32) * 0.125
    sess = tf.Session()
    uv_val = sess.run(flowToColor(uv, 0.1))
    
    import scipy.misc
    scipy.misc.imshow(uv_val[0, :, :, :])
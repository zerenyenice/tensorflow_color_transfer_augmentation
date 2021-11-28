import tensorflow as tf
import numpy as np
import os


# tensorflow color transfer augmentation
def load_stats(path):
    try:
        file = np.load(os.path.join(path, 'ct.npy'))
    except FileNotFoundError:
        file = None
    return file


def color_transfer_tf(source, target, source_mask=None,target_mask=None, preserve_paper=True, normalize=True):
    if normalize:
        source = source / 255.0
        target = target / 255.0
        if source_mask is not None:
            source_mask = source_mask / 255.0
            target_mask = target_mask / 255.0

    source = bgr2lab(source)

    l, a, b = tf.unstack(source, axis=-1)

    (l_m_src, l_std_src, a_m_src, a_std_src, b_m_src, b_std_src) = lab_stats_tf(source, mask=source_mask)
    (l_m_trgt, l_std_trgt, a_m_trgt, a_std_trgt, b_m_trgt, b_std_trgt) = lab_stats_tf(target, mask=target_mask)

    l -= l_m_src
    a -= a_m_src
    b -= b_m_src

    if preserve_paper:
        l = (l_std_trgt / l_std_src) * l
        a = (a_std_trgt / a_std_src) * a
        b = (b_std_trgt / b_std_src) * b
    else:
        l = (l_std_src / l_std_trgt) * l
        a = (a_std_src / a_std_trgt) * a
        b = (b_std_src / b_std_trgt) * b

    l += l_m_trgt
    a += a_m_trgt
    b += b_m_trgt

    l = tf.clip_by_value(l, 0, 100)
    a = tf.clip_by_value(a, -127, 127)
    b = tf.clip_by_value(b, -127, 127)

    transfer = tf.stack([l, a, b], axis=-1)
    transfer = lab2bgr(transfer)
    if normalize:
        transfer = tf.clip_by_value(transfer, 0, 1)
        transfer = transfer * 255.0
    else:
        transfer = tf.clip_by_value(transfer, 0, 255)

    return transfer


@tf.function
def bgr2lab(sbgr):
    with tf.name_scope('rgb_to_lab'):
        sbgr = check_image(sbgr)
        sbgr_pixels = tf.where(sbgr > 0.04045, ((sbgr + 0.055) / 1.055) ** 2.4, sbgr / 12.92)
        b, g, r = tf.unstack(sbgr_pixels, axis=-1)
        x = 0.412453 * r + 0.357580 * g + 0.180423 * b
        y = 0.212671 * r + 0.715160 * g + 0.072169 * b
        z = 0.019334 * r + 0.119193 * g + 0.950227 * b
        x = x / 0.950456
        z = z / 1.088754
        fx = tf.where(x > 0.008856, x ** (1 / 3), 7.787 * x + 16. / 116.)
        fy = tf.where(x > 0.008856, y ** (1 / 3), 7.787 * y + 16. / 116.)
        fz = tf.where(x > 0.008856, z ** (1 / 3), 7.787 * z + 16. / 116.)
        l = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        lab = tf.stack([l, a, b], axis=-1)
        return lab


@tf.function
def lab2bgr(lab):
    with tf.name_scope('lab_to_rgb'):
        lab = check_image(lab)
        l, a, b = tf.unstack(lab, axis=-1)
        fy = (l + 16.) / 116.
        fx = (a / 500.) + fy
        fz = fy - (b / 200.)
        x = tf.where(fx > 0.2068966, fx ** 3, (fx - 16. / 166.) / 7.787)
        y = tf.where(fy > 0.2068966, fy ** 3, (fy - 16. / 166.) / 7.787)
        z = tf.where(fz > 0.2068966, fz ** 3, (fz - 16. / 166.) / 7.787)
        x = x * 0.950456
        z = z * 1.088754
        r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
        g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
        b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
        bgr = tf.stack([b, g, r], axis=-1)
        bgr = tf.clip_by_value(bgr, 0., 1.)
        bgr = tf.where(bgr > 0.0031308, 1.055 * (bgr ** (1 / 2.4)) - 0.055, bgr * 12.92)
        return bgr


@tf.function
def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message='image must have 3 color channels')
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)
    if image.get_shape().ndims not in (3, 4):
        raise ValueError('image must be either 3 or 4 dimensions')
    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


@tf.function
def lab_stats_tf(batch, mask=None):
    # compute the mean and standard deviation of each channel
    l_b, a_b, b_b = tf.unstack(batch, axis=-1)
    if mask is not None:
        mask_b = tf.squeeze(mask, axis=-1)
    else:
        mask_b = tf.ones_like(l_b)
    l_mean, l_var = tf.nn.weighted_moments(l_b, (0, 1), mask_b, keepdims=True)
    l_std = tf.sqrt(l_var)
    a_mean, a_var = tf.nn.weighted_moments(a_b, (0, 1), mask_b, keepdims=True)
    a_std = tf.sqrt(a_var)
    b_mean, b_var = tf.nn.weighted_moments(b_b, (0, 1), mask_b, keepdims=True)
    b_std = tf.sqrt(b_var)
    # return the color statistics
    return l_mean, l_std, a_mean, a_std, b_mean, b_std


def generate_stats(output, face_mask):
    output = output / 255.0
    face_mask = face_mask / 255.0

    lab = bgr2lab(output)
    stats = lab_stats_tf(lab, face_mask)
    return stats
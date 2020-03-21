# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.configs import cfgs
from libs.networks.resnet import fusion_two_layer, add_heatmap, generate_mask, enlarge_RF

DATA_FORMAT = "NHWC"
DEBUG = False
debug_dict = {}
BottleNeck_NUM_DICT = {
    'resnet50_v1b': [3, 4, 6, 3],
    'resnet101_v1b': [3, 4, 23, 3],
    'resnet152_v1b': [3, 8, 36, 3],
    'resnet50_v1d': [3, 4, 6, 3],
    'resnet101_v1d': [3, 4, 23, 3],
    'resnet152_v1d': [3, 8, 36, 3]
}

BASE_CHANNELS_DICT = {
    'resnet50_v1b': [64, 128, 256, 512],
    'resnet101_v1b': [64, 128, 256, 512],
    'resnet152_v1b': [64, 128, 256, 512],
    'resnet50_v1d': [64, 128, 256, 512],
    'resnet101_v1d': [64, 128, 256, 512],
    'resnet152_v1d': [64, 128, 256, 512]
}


def resnet_arg_scope(freeze_norm, is_training=True, weight_decay=0.0001,
                     batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True):

    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'data_format': DATA_FORMAT
    }
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


def stem_7x7(net, scope="C1"):

    with tf.variable_scope(scope):
        net = tf.pad(net, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]])  # pad for data
        net = slim.conv2d(net, num_outputs=64, kernel_size=[7, 7], stride=2,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope="conv0")
        if DEBUG:
            debug_dict['conv_7x7_bn_relu'] = tf.transpose(net, [0, 3, 1, 2])  # NHWC --> NCHW
        net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding="VALID", data_format=DATA_FORMAT)
        return net


def stem_stack_3x3(net, input_channel=32, scope="C1"):
    with tf.variable_scope(scope):
        net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        net = slim.conv2d(net, num_outputs=input_channel, kernel_size=[3, 3], stride=2,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv0')
        net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        net = slim.conv2d(net, num_outputs=input_channel, kernel_size=[3, 3], stride=1,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv1')
        net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        net = slim.conv2d(net, num_outputs=input_channel*2, kernel_size=[3, 3], stride=1,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv2')
        net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding="VALID", data_format=DATA_FORMAT)
        return net


def bottleneck_v1b(input_x, base_channel, scope, stride=1, projection=False, avg_down=True):
    '''
    for bottleneck_v1b: reduce spatial dim in conv_3x3 with stride 2.
    '''
    with tf.variable_scope(scope):
        if DEBUG:
            debug_dict[input_x.op.name] = tf.transpose(input_x, [0, 3, 1, 2])
        net = slim.conv2d(input_x, num_outputs=base_channel, kernel_size=[1, 1], stride=1,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv0')
        if DEBUG:
            debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])

        net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        if DEBUG:
            debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])

        net = slim.conv2d(net, num_outputs=base_channel, kernel_size=[3, 3], stride=stride,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          scope='conv1')
        if DEBUG:
            debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])
        net = slim.conv2d(net, num_outputs=base_channel * 4, kernel_size=[1, 1], stride=1,
                          padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                          activation_fn=None, scope='conv2')
        if DEBUG:
            debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])
        # Note that : gamma in the last conv should be init with 0.
        # But we just reload params from mxnet, so don't specific batch norm initializer
        if projection:

            if avg_down:  # design for resnet_v1d
                '''
                In GluonCV, padding is "ceil mode". Here we use "SAME" to replace it, which may cause Erros.
                And the erro will grow with depth of resnet. e.g. res101 erro > res50 erro
                '''
                shortcut = slim.avg_pool2d(input_x, kernel_size=[stride, stride], stride=stride, padding="SAME",
                                           data_format=DATA_FORMAT)
                if DEBUG:
                    debug_dict[shortcut.op.name] = tf.transpose(shortcut, [0, 3, 1, 2])

                shortcut = slim.conv2d(shortcut, num_outputs=base_channel*4, kernel_size=[1, 1],
                                       stride=1, padding="VALID", biases_initializer=None, data_format=DATA_FORMAT,
                                       activation_fn=None,
                                       scope='shortcut')
                if DEBUG:
                    debug_dict[shortcut.op.name] = tf.transpose(shortcut, [0, 3, 1, 2])
                # shortcut should have batch norm.
            else:
                shortcut = slim.conv2d(input_x, num_outputs=base_channel * 4, kernel_size=[1, 1],
                                       stride=stride, padding="VALID", biases_initializer=None, activation_fn=None,
                                       data_format=DATA_FORMAT,
                                       scope='shortcut')
                if DEBUG:
                    debug_dict[shortcut.op.name] = tf.transpose(shortcut, [0, 3, 1, 2])
        else:
            shortcut = tf.identity(input_x, name='shortcut/Identity')
            if DEBUG:
                debug_dict[shortcut.op.name] = tf.transpose(shortcut, [0, 3, 1, 2])

        net = net + shortcut
        if DEBUG:
            debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])
        net = tf.nn.relu(net)
        if DEBUG:
            debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])
        return net


def make_block(net, base_channel, bottleneck_nums, scope, avg_down=True, spatial_downsample=False):
    with tf.variable_scope(scope):
        first_stride = 2 if spatial_downsample else 1

        net = bottleneck_v1b(input_x=net, base_channel=base_channel,scope='bottleneck_0',
                             stride=first_stride, avg_down=avg_down, projection=True)
        for i in range(1, bottleneck_nums):
            net = bottleneck_v1b(input_x=net, base_channel=base_channel, scope="bottleneck_%d" % i,
                                 stride=1, avg_down=avg_down, projection=False)
        return net


def get_resnet_v1_b_base(input_x, freeze_norm, scope="resnet50_v1b", bottleneck_nums=[3, 4, 6, 3], base_channels=[64, 128, 256, 512],
                    freeze=[True, False, False, False, False], is_training=True):

    assert len(bottleneck_nums) == len(base_channels), "bottleneck num should same as base_channels size"
    assert len(freeze) == len(bottleneck_nums) +1, "should satisfy:: len(freeze) == len(bottleneck_nums) + 1"
    feature_dict = {}
    with tf.variable_scope(scope):
        with slim.arg_scope(resnet_arg_scope(is_training=(not freeze[0]) and is_training,
                                             freeze_norm=freeze_norm)):
            net = stem_7x7(net=input_x, scope="C1")
            feature_dict["C1"] = net
        for i in range(2, len(bottleneck_nums)+2):
            spatial_downsample = False if i == 2 else True
            with slim.arg_scope(resnet_arg_scope(is_training=(not freeze[i-1]) and is_training,
                                                 freeze_norm=freeze_norm)):
                net = make_block(net=net, base_channel=base_channels[i-2],
                                 bottleneck_nums=bottleneck_nums[i-2],
                                 scope="C%d" % i,
                                 avg_down=False, spatial_downsample=spatial_downsample)
                feature_dict["C%d" % i] = net

    return net, feature_dict


def get_resnet_v1_d_base(input_x, freeze_norm, scope="resnet50_v1d", bottleneck_nums=[3, 4, 6, 3], base_channels=[64, 128, 256, 512],
                    freeze=[True, False, False, False, False], is_training=True):

    assert len(bottleneck_nums) == len(base_channels), "bottleneck num should same as base_channels size"
    assert len(freeze) == len(bottleneck_nums) + 1, "should satisfy:: len(freeze) == len(bottleneck_nums) + 1"
    feature_dict = {}
    with tf.variable_scope(scope):
        with slim.arg_scope(resnet_arg_scope(is_training=((not freeze[0]) and is_training),
                                             freeze_norm=freeze_norm)):
            net = stem_stack_3x3(net=input_x, input_channel=32, scope="C1")
            feature_dict["C1"] = net
            # print (net)
        for i in range(2, len(bottleneck_nums)+2):
            spatial_downsample = False if i == 2 else True  # do not downsample in C2
            with slim.arg_scope(resnet_arg_scope(is_training=((not freeze[i-1]) and is_training),
                                                 freeze_norm=freeze_norm)):
                net = make_block(net=net, base_channel=base_channels[i-2],
                                 bottleneck_nums=bottleneck_nums[i-2],
                                 scope="C%d" % i,
                                 avg_down=True, spatial_downsample=spatial_downsample)
                feature_dict["C%d" % i] = net

    return net, feature_dict


# -----------------------------------
def resnet_base(img_batch, scope_name, is_training=True):
    if scope_name.endswith('b'):
        get_resnet_fn = get_resnet_v1_b_base
    elif scope_name.endswith('d'):
        get_resnet_fn = get_resnet_v1_d_base
    else:
        raise ValueError("scope Name erro....")

    _, feature_dict = get_resnet_fn(input_x=img_batch, scope=scope_name,
                                    bottleneck_nums=BottleNeck_NUM_DICT[scope_name],
                                    base_channels=BASE_CHANNELS_DICT[scope_name],
                                    is_training=is_training, freeze_norm=True,
                                    freeze=cfgs.FREEZE_BLOCKS)

    # for level in range(5, 1, -1):
    #     add_heatmap(feature_dict['C%d' % level], name='Layer%d/C%d_heat' % (level, level))
        # feat_dict['C%d' %level] = tf.Print(feat_dict['C%d' %level], [tf.shape(feat_dict['C%d' %level])],
        #                                    message='C%d shape-->'% level, summarize=10)

    pyramid_dict = {}
    with tf.variable_scope('build_pyramid'):
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY),
                            activation_fn=None, normalizer_fn=None):

            P5 = slim.conv2d(feature_dict['C5'],
                             num_outputs=256,
                             kernel_size=[1, 1],
                             stride=1, scope='build_P5')

            if cfgs.ADD_GLOBAL_CTX:
                print(10*"ADD GLOBAL CTX.....")
                global_ctx = tf.reduce_mean(feature_dict['C5'], axis=[1, 2], keep_dims=True)
                global_ctx = slim.conv2d(global_ctx, kernel_size=[1, 1], num_outputs=256, stride=1,
                                         activation_fn=None, scope='global_ctx')
                pyramid_dict['P5'] = P5 + global_ctx
            else:
                pyramid_dict['P5'] = P5

            for level in range(4, 1, -1):  # build [P4, P3, P2]

                pyramid_dict['P%d' % level] = fusion_two_layer(C_i=feature_dict["C%d" % level],
                                                               P_j=pyramid_dict["P%d" % (level + 1)],
                                                               scope='build_P%d' % level)
            for level in range(5, 1, -1):  # use 3x3 conv fuse P5, P4, P3, P2
                pyramid_dict['P%d' % level] = slim.conv2d(pyramid_dict['P%d' % level],
                                                          num_outputs=256, kernel_size=[3, 3], padding="SAME",
                                                          stride=1, scope="fuse_P%d" % level)
            if (not cfgs.USE_SUPERVISED_MASK) and "P6" in cfgs.LEVLES:
                # if use supervised_mask, we get p6 after enlarge RF
                pyramid_dict['P6'] = slim.avg_pool2d(pyramid_dict["P5"], kernel_size=[2, 2],
                                                     stride=2, scope='build_P6')
    for level in range(5, 1, -1):
        add_heatmap(pyramid_dict['P%d' % level], name='Layer%d/P%d_fpn_heat' % (level, level))

    if not cfgs.USE_SUPERVISED_MASK:
        print("we are in Pyramid::-======>>>>")
        print(cfgs.LEVLES)
        print("base_anchor_size are: ", cfgs.BASE_ANCHOR_SIZE_LIST)
        print(20 * "__")
        return [pyramid_dict[level_name] for level_name in cfgs.LEVLES]

    else:
        mask_list = []
        with tf.variable_scope("enrich_semantics"):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY),
                                normalizer_fn=None):
                for i, l_name in enumerate(cfgs.GENERATE_MASK_LIST):
                    G, mask, dot_layer = generate_mask(net=pyramid_dict[l_name],
                                                       num_layer=cfgs.ADDITION_LAYERS[i],
                                                       level_name=l_name)
                    add_heatmap(G, name="MASK/G_%s" % l_name)
                    add_heatmap(mask, name="MASK/mask_%s" % l_name)

                    if cfgs.MASK_ACT_FET:
                        pyramid_dict[l_name] = pyramid_dict[l_name] * dot_layer
                    mask_list.append(mask)

        with tf.variable_scope("enlarge_RF"):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY),
                                normalizer_fn=None):
                for i, l_name in enumerate(cfgs.ENLAEGE_RF_LIST):
                    pyramid_dict[l_name] = enlarge_RF(net=pyramid_dict[l_name],
                                                      num_layer=2, k_size=3, rate=2)
                if "P6" in cfgs.LEVLES:
                    pyramid_dict['P6'] = slim.avg_pool2d(pyramid_dict["P5"], kernel_size=[2, 2],
                                                         stride=2, scope='build_P6')
                    pyramid_dict["P6"] = slim.conv2d(pyramid_dict["P6"],
                                                     num_outputs=256, kernel_size=[3, 3], stride=1, rate=2)
        for level in range(5, 1, -1):
            add_heatmap(pyramid_dict['P%d' % level], name='Layer%d/P%d_lastheat' % (level, level))

        # return [P2, P3, P4, P5, P6]
        print("we are in Pyramid::-======>>>>")
        print(cfgs.LEVLES)
        print("base_anchor_size are: ", cfgs.BASE_ANCHOR_SIZE_LIST)
        print(20 * "__")
        return [pyramid_dict[level_name] for level_name in cfgs.LEVLES], mask_list



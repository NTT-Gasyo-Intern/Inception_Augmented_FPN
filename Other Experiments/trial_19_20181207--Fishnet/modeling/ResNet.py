# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Implements ResNet and ResNeXt.

See: https://arxiv.org/abs/1512.03385, https://arxiv.org/abs/1611.05431.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.net import get_group_gn

import collections
import numpy as np

from detectron.core.config import cfg
from detectron.modeling.generate_anchors import generate_anchors
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils

# ---------------------------------------------------------------------------- #
# Bits for specific architectures (ResNet50, ResNet101, ...)
# ---------------------------------------------------------------------------- #


def add_ResNet50_conv4_body(model):
    return add_ResNet_convX_body(model, (3, 4, 6))


# def add_ResNet50_conv5_body(model):
#     return add_ResNet_convX_body(model, (3, 4, 6, 3))


def add_ResNet101_conv4_body(model):
    return add_ResNet_convX_body(model, (3, 4, 23))


def add_ResNet101_conv5_body(model):
    return add_ResNet_convX_body(model, (3, 4, 23, 3))


def add_ResNet152_conv5_body(model):
    return add_ResNet_convX_body(model, (3, 8, 36, 3))

def add_ResNet50_conv5_body(model):
    return add_ResNet_convX_body(
        model,
        (65, 128, 256, 512, 512, 512, 256, 256, 320, 832, 1600), # channel
        (2, 4, 8, 4, 2, 2, 2, 2, 2, 4), # num_res_blk
        (2, 2, 2, 2, 2, 4) # num_trans_blk
)

# ---------------------------------------------------------------------------- #
# Generic ResNet components
# ---------------------------------------------------------------------------- #


def add_stage(
    model,
    prefix,
    blob_in,
    n,
    dim_in,
    dim_out,
    dim_inner,
    dilation,
    stride_init=2
):
    """Add a ResNet stage to the model by stacking n residual blocks."""
    # e.g., prefix = res2
    for i in range(n):
        blob_in = add_residual_block(
            model,
            '{}_{}'.format(prefix, i),
            blob_in,
            dim_in,
            dim_out,
            dim_inner,
            dilation,
            stride_init,
            # Not using inplace for the last block;
            # it may be fetched externally or used by FPN
            inplace_sum=i < n - 1
        )
        dim_in = dim_out
    return blob_in, dim_in


def add_ResNet_convX_body(model, channel_counts, res_counts, trans_counts):
    """FishNet Trial."""

    # freeze_at = cfg.TRAIN.FREEZE_AT
    # assert freeze_at in [0, 2, 3, 4, 5]

    # add the stem (by default, conv1 and pool1 with bn; can support gn)
    p, dim_in = globals()[cfg.RESNETS.STEM_FUNC](model, 'data')

    dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP

    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11) = channel_counts[:11]
    (rn1, rn2, rn3, rn4, rn5, rn6, rn7, rn8, rn9, rn10) = res_counts[:10]
    (tn1, tn2, tn3, tn4, tn5, tn6) = trans_counts[:6]

    #
    # Create Tail
    #

    # TODO: After pre-trained weights is available, follow 
    #       FishNet150 strictly

    tail_2, dim_in = add_stage(
        model, 'res2', p, rn1, dim_in, c1, int(c1 / 4), 1
    )

    tail_3, dim_in = add_stage(
        model, 'res3', tail_2, rn2, dim_in, c2, int(c2 / 4), 1
    )

    tail_4, dim_in = add_stage(
        model, 'res4', tail_3, rn3, dim_in, c3, int(c3 / 4), 1
    )

    tail_5, dim_in = add_stage(
        model, 'res5', tail_4, rn4, dim_in, c4, int(c4 / 4), 1
    )

    #
    # Create Tail->Body Trans
    #

    trans_1, dim_in = add_stage(
        model, 'trans1', 'res4_0_branch1', tn1, c3, int(c3 / 2), int(c3 / 8), 1, 1
    )

    trans_2, dim_in = add_stage(
        model, 'trans2', 'res3_0_branch1', tn2, c2, int(c2 / 2), int(c2 / 8), 1, 1
    )

    trans_3, dim_in = add_stage(
        model, 'trans3', 'res2_0_branch1', tn3, c1, int(c1 / 2), int(c1 / 8), 1, 1
    )

    #
    # Create Body
    #

    ################################ MODULE ###################################
    tail_5_2x = model.net.UpsampleNearest(tail_5, tail_5 + '_2x', scale=2)

    body_4_cc, _ = model.net.Concat(
        [trans_1, tail_5_2x],
        ['body_4_cc','body_4_cc_dim'],
        order = 'NCHW'
    )

    # r(.)
    body_4_branch_1x1 = model.Conv(
        body_4_cc,
        'body_4_branch_1x1',
        dim_in=int(c4 + c3 / 2),
        dim_out=c5,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0)
    )
    body_4_branch_1x1 = model.Relu(body_4_branch_1x1, body_4_branch_1x1)

    # M(.)
    body_4, dim_in = add_stage(
        model, 'body4', body_4_cc, rn5, int(c4 + c3 / 2), c5, int(c5 / 4), 1, 1
    )

    body_4_out = model.net.Sum([body_4, body_4_branch_1x1], 'body_4_out') # dim c5
    ###########################################################################

    ################################ MODULE ###################################
    body_4_out_2x = model.net.UpsampleNearest(body_4_out, body_4_out + '_2x', scale=2)

    body_3_cc, _ = model.net.Concat(
        [trans_2, body_4_out_2x],
        ['body_3_cc','body_3_cc_dim'],
        order = 'NCHW'
    )

    # r(.)
    body_3_branch_1x1 = model.Conv(
        body_3_cc,
        'body_3_branch_1x1',
        dim_in=int(c5 + c2 / 2),
        dim_out=c6,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0)
    )
    body_3_branch_1x1 = model.Relu(body_3_branch_1x1, body_3_branch_1x1)

    # M(.)
    body_3, dim_in = add_stage(
        model, 'body3', body_3_cc, rn6, int(c5 + c2 / 2), c6, int(c6 / 4), 1, 1
    )

    body_3_out = model.net.Sum([body_3, body_3_branch_1x1], 'body_3_out') # dim c6
    ###########################################################################

    ################################ MODULE ###################################
    body_3_out_2x = model.net.UpsampleNearest(body_3_out, body_3_out + '_2x', scale=2)

    body_2_cc, _ = model.net.Concat(
        [trans_3, body_3_out_2x],
        ['body_2_cc','body_2_cc_dim'],
        order = 'NCHW'
    )

    # r(.)
    body_2_branch_1x1 = model.Conv(
        body_2_cc,
        'body_2_branch_1x1',
        dim_in=int(c6 + c1 / 2),
        dim_out=c7,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0)
    )
    body_2_branch_1x1 = model.Relu(body_2_branch_1x1, body_2_branch_1x1)

    # M(.)
    body_2, dim_in = add_stage(
        model, 'body2', body_2_cc, rn7, int(c6 + c1 / 2), c7, int(c7 / 4), 1, 1
    )

    body_2_out = model.net.Sum([body_2, body_2_branch_1x1], 'body_2_out') # dim c7
    ###########################################################################

    #
    # Create Body->Head Trans
    #

    trans_4, dim_in = add_stage(
        model, 'trans4', body_3_cc, tn4, int(c5 + c2 / 2), int((c5 + c2 / 2) / 2), int((c5 + c2 / 2) / 8), 1, 1 # 3
    )

    trans_5, dim_in = add_stage(
        model, 'trans5', body_4_cc, tn5, int(c4 + c3 / 2), int((c4 + c3 / 2) / 2), int((c4 + c3 / 2) / 8), 1, 1 # 4
    )

    trans_6, dim_in = add_stage(
        model, 'trans6', 'res5_0_branch1', tn6, c4, int(c4 / 2), int(c4 / 8), 1, 1 # 5
    )

    # 
    # Create Head
    #

    ################################ MODULE ###################################
    body_2_out_DS = model.MaxPool(
        body_2_out,
        'body_2_out_DS',
        kernel=2,
        pad=0,
        stride=2
    )

    head_3_cc, _ = model.net.Concat(
        [trans_4, body_2_out_DS],
        ['head_3_cc','head_3_cc_dim'],
        order = 'NCHW' # c = c7 + c5 + c2
    )

    head_3_cc_1x1 = model.Conv(
        head_3_cc,
        'head_3_cc_1x1',
        dim_in=int(c7 + ((c5 + c2 / 2) / 2)),
        dim_out=c8,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0)
    )
    head_3_cc_1x1 = model.Relu(head_3_cc_1x1, head_3_cc_1x1)

    # M(.)
    head_3, dim_in = add_stage(
        model, 'head3', head_3_cc_1x1, rn8, c8, c8, int(c8 / 4), 1, 1
    )

    head_3_out = model.net.Sum([head_3, head_3_cc_1x1], 'head_3_out') # dim c8
    ###########################################################################

    ################################ MODULE ###################################
    head_3_out_DS = model.MaxPool(
        head_3_out,
        'head_3_out_DS',
        kernel=2,
        pad=0,
        stride=2
    )

    head_4_cc, _ = model.net.Concat(
        [trans_5, head_3_out_DS],
        ['head_4_cc','head_4_cc_dim'],
        order = 'NCHW' # c = c8 + c4 + c3
    )

    head_4_cc_1x1 = model.Conv(
        head_4_cc,
        'head_4_cc_1x1',
        dim_in=int(c8 + ((c4 + c3 / 2) / 2)),
        dim_out=c9,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0)
    )
    head_4_cc_1x1 = model.Relu(head_4_cc_1x1, head_4_cc_1x1)

    # M(.)
    head_4, dim_in = add_stage(
        model, 'head4', head_4_cc_1x1, rn9, c9, c9, int(c9 / 4), 1, 1
    )

    head_4_out = model.net.Sum([head_4, head_4_cc_1x1], 'head_4_out') # dim c9
    ###########################################################################

    ################################ MODULE ###################################
    head_4_out_DS = model.MaxPool(
        head_4_out,
        'head_4_out_DS',
        kernel=2,
        pad=0,
        stride=2
    )

    head_5_cc, _ = model.net.Concat(
        [trans_6, head_4_out_DS],
        ['head_5_cc','head_5_cc_dim'],
        order = 'NCHW' # c = c9 + c4
    )

    head_5_cc_1x1 = model.Conv(
        head_5_cc,
        'head_5_cc_1x1',
        dim_in=int(c9 + c4 / 2),
        dim_out=c10,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0)
    )
    head_5_cc_1x1 = model.Relu(head_5_cc_1x1, head_5_cc_1x1)

    # M(.)
    head_5, dim_in = add_stage(
        model, 'head5', head_5_cc_1x1, rn10, c10, c10, int(c10 / 4), 1, 1
    )

    head_5_out = model.net.Sum([head_5, head_5_cc_1x1], 'head_5_out') # dim c10
    ###########################################################################


    return head_5_out, c10, 1. / 32.






def add_ResNet_convX_body_work(model, block_counts):
    """Add a ResNet body from input data up through the res5 (aka conv5) stage.
    The final res5/conv5 stage may be optionally excluded (hence convX, where
    X = 4 or 5)."""
    freeze_at = cfg.TRAIN.FREEZE_AT
    assert freeze_at in [0, 2, 3, 4, 5]

    # add the stem (by default, conv1 and pool1 with bn; can support gn)
    p, dim_in = globals()[cfg.RESNETS.STEM_FUNC](model, 'data')

    dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
    (n1, n2, n3) = block_counts[:3]
    s, dim_in = add_stage(model, 'res2', p, n1, dim_in, 256, dim_bottleneck, 1)
    if freeze_at == 2:
        model.StopGradient(s, s)
    s, dim_in = add_stage(
        model, 'res3', s, n2, dim_in, 128, dim_bottleneck * 2, 1
    )
    if freeze_at == 3:
        model.StopGradient(s, s)
    s, dim_in = add_stage(
        model, 'res4', s, n3, dim_in, 256, dim_bottleneck * 4, 1
    )
    if freeze_at == 4:
        model.StopGradient(s, s)
    if len(block_counts) == 4:
        n4 = block_counts[3]
        s, dim_in = add_stage(
            model, 'res5', s, n4, dim_in, 512, dim_bottleneck * 8,
            cfg.RESNETS.RES5_DILATION
        )
        if freeze_at == 5:
            model.StopGradient(s, s)
        return s, dim_in, 1. / 32. * cfg.RESNETS.RES5_DILATION
    else:
        return s, dim_in, 1. / 16.


def add_ResNet_roi_conv5_head(model, blob_in, dim_in, spatial_scale):
    """Adds an RoI feature transformation (e.g., RoI pooling) followed by a
    res5/conv5 head applied to each RoI."""
    # TODO(rbg): This contains Fast R-CNN specific config options making it non-
    # reusable; make this more generic with model-specific wrappers
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
    stride_init = int(cfg.FAST_RCNN.ROI_XFORM_RESOLUTION / 7)
    s, dim_in = add_stage(
        model, 'res5', 'pool5', 3, dim_in, 2048, dim_bottleneck * 8, 1,
        stride_init
    )
    s = model.AveragePool(s, 'res5_pool', kernel=7)
    return s, 2048


def add_residual_block(
    model,
    prefix,
    blob_in,
    dim_in,
    dim_out,
    dim_inner,
    dilation,
    stride_init=2,
    inplace_sum=False
):
    """Add a residual block to the model."""
    # prefix = res<stage>_<sub_stage>, e.g., res2_3

    # Max pooling is performed prior to the first stage (which is uniquely
    # distinguished by dim_in = 64), thus we keep stride = 1 for the first stage
    stride = stride_init if (
        dim_in != dim_out and dim_in != 64 and dilation == 1
    ) else 1

    # transformation blob
    tr = globals()[cfg.RESNETS.TRANS_FUNC](
        model,
        blob_in,
        dim_in,
        dim_out,
        stride,
        prefix,
        dim_inner,
        group=cfg.RESNETS.NUM_GROUPS,
        dilation=dilation
    )

    # sum -> ReLU
    # shortcut function: by default using bn; support gn
    add_shortcut = globals()[cfg.RESNETS.SHORTCUT_FUNC]
    sc = add_shortcut(model, prefix, blob_in, dim_in, dim_out, stride)
    if inplace_sum:
        s = model.net.Sum([tr, sc], tr)
    else:
        s = model.net.Sum([tr, sc], prefix + '_sum')

    return model.Relu(s, s)

# ------------------------------------------------------------------------------
# various shortcuts (may expand and may consider a new helper)
# ------------------------------------------------------------------------------


def basic_bn_shortcut(model, prefix, blob_in, dim_in, dim_out, stride):
    """ For a pre-trained network that used BN. An AffineChannel op replaces BN
    during fine-tuning.
    """

    if dim_in == dim_out:
        return blob_in

    c = model.Conv(
        blob_in,
        prefix + '_branch1',
        dim_in,
        dim_out,
        kernel=1,
        stride=stride,
        no_bias=1
    )
    return model.AffineChannel(c, prefix + '_branch1_bn', dim=dim_out)


def basic_gn_shortcut(model, prefix, blob_in, dim_in, dim_out, stride):
    if dim_in == dim_out:
        return blob_in

    # output name is prefix + '_branch1_gn'
    return model.ConvGN(
        blob_in,
        prefix + '_branch1',
        dim_in,
        dim_out,
        kernel=1,
        group_gn=get_group_gn(dim_out),
        stride=stride,
        pad=0,
        group=1,
    )


# ------------------------------------------------------------------------------
# various stems (may expand and may consider a new helper)
# ------------------------------------------------------------------------------


def basic_bn_stem(model, data, **kwargs):
    """Add a basic ResNet stem. For a pre-trained network that used BN.
    An AffineChannel op replaces BN during fine-tuning.
    """

    dim = 64
    p = model.Conv(data, 'conv1', 3, dim, 7, pad=3, stride=2, no_bias=1)
    p = model.AffineChannel(p, 'res_conv1_bn', dim=dim, inplace=True)
    p = model.Relu(p, p)
    p = model.MaxPool(p, 'pool1', kernel=3, pad=1, stride=2)
    return p, dim


def basic_gn_stem(model, data, **kwargs):
    """Add a basic ResNet stem (using GN)"""

    dim = 64
    p = model.ConvGN(
        data, 'conv1', 3, dim, 7, group_gn=get_group_gn(dim), pad=3, stride=2
    )
    p = model.Relu(p, p)
    p = model.MaxPool(p, 'pool1', kernel=3, pad=1, stride=2)
    return p, dim


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------


def bottleneck_transformation(
    model,
    blob_in,
    dim_in,
    dim_out,
    stride,
    prefix,
    dim_inner,
    dilation=1,
    group=1
):
    """Add a bottleneck transformation to the model."""
    # In original resnet, stride=2 is on 1x1.
    # In fb.torch resnet, stride=2 is on 3x3.
    (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)

    # conv 1x1 -> BN -> ReLU
    cur = model.ConvAffine(
        blob_in,
        prefix + '_branch2a',
        dim_in,
        dim_inner,
        kernel=1,
        stride=str1x1,
        pad=0,
        inplace=True
    )
    cur = model.Relu(cur, cur)

    # conv 3x3 -> BN -> ReLU
    cur = model.ConvAffine(
        cur,
        prefix + '_branch2b',
        dim_inner,
        dim_inner,
        kernel=3,
        stride=str3x3,
        pad=1 * dilation,
        dilation=dilation,
        group=group,
        inplace=True
    )
    cur = model.Relu(cur, cur)

    # conv 1x1 -> BN (no ReLU)
    # NB: for now this AffineChannel op cannot be in-place due to a bug in C2
    # gradient computation for graphs like this
    cur = model.ConvAffine(
        cur,
        prefix + '_branch2c',
        dim_inner,
        dim_out,
        kernel=1,
        stride=1,
        pad=0,
        inplace=False
    )
    return cur


def bottleneck_gn_transformation(
    model,
    blob_in,
    dim_in,
    dim_out,
    stride,
    prefix,
    dim_inner,
    dilation=1,
    group=1
):
    """Add a bottleneck transformation with GroupNorm to the model."""
    # In original resnet, stride=2 is on 1x1.
    # In fb.torch resnet, stride=2 is on 3x3.
    (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)

    # conv 1x1 -> GN -> ReLU
    cur = model.ConvGN(
        blob_in,
        prefix + '_branch2a',
        dim_in,
        dim_inner,
        kernel=1,
        group_gn=get_group_gn(dim_inner),
        stride=str1x1,
        pad=0,
    )
    cur = model.Relu(cur, cur)

    # conv 3x3 -> GN -> ReLU
    cur = model.ConvGN(
        cur,
        prefix + '_branch2b',
        dim_inner,
        dim_inner,
        kernel=3,
        group_gn=get_group_gn(dim_inner),
        stride=str3x3,
        pad=1 * dilation,
        dilation=dilation,
        group=group,
    )
    cur = model.Relu(cur, cur)

    # conv 1x1 -> GN (no ReLU)
    cur = model.ConvGN(
        cur,
        prefix + '_branch2c',
        dim_inner,
        dim_out,
        kernel=1,
        group_gn=get_group_gn(dim_out),
        stride=1,
        pad=0,
    )
    return cur

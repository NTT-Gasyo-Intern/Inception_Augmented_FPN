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

"""Various network "heads" for predicting masks in Mask R-CNN.

The design is as follows:

... -> RoI ----\
                -> RoIFeatureXform -> mask head -> mask output -> loss
... -> Feature /
       Map

The mask head produces a feature representation of the RoI for the purpose
of mask prediction. The mask output module converts the feature representation
into real-valued (soft) masks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.modeling.ResNet as ResNet
import detectron.utils.blob as blob_utils


import logging
import os
import sys
import yaml


# ---------------------------------------------------------------------------- #
# Mask R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_mask_rcnn_outputs(model, blob_in, dim):
    """Add Mask R-CNN specific outputs: either mask logits or probs."""
    num_cls = cfg.MODEL.NUM_CLASSES if cfg.MRCNN.CLS_SPECIFIC_MASK else 1

    if cfg.MRCNN.USE_FC_OUTPUT:
        # Predict masks with a fully connected layer (ignore 'fcn' in the blob
        # name)
        blob_out = model.FC(
            blob_in,
            'mask_fcn_logits',
            dim,
            num_cls * cfg.MRCNN.RESOLUTION**2,
            weight_init=gauss_fill(0.001),
            bias_init=const_fill(0.0)
        )
    else:
        # Predict mask using Conv

        # Use GaussianFill for class-agnostic mask prediction; fills based on
        # fan-in can be too large in this case and cause divergence
        fill = (
            cfg.MRCNN.CONV_INIT
            if cfg.MRCNN.CLS_SPECIFIC_MASK else 'GaussianFill'
        )
        blob_out = model.Conv(
            blob_in,
            'mask_fcn_logits',
            dim,
            num_cls,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(fill, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )

        if cfg.MRCNN.UPSAMPLE_RATIO > 1:
            blob_out = model.BilinearInterpolation(
                'mask_fcn_logits', 'mask_fcn_logits_up', num_cls, num_cls,
                cfg.MRCNN.UPSAMPLE_RATIO
            )

    if not model.train:  # == if test
        blob_out = model.net.Sigmoid(blob_out, 'mask_fcn_probs')

    return blob_out


def add_mask_rcnn_losses(model, blob_mask):
    """Add Mask R-CNN specific losses."""
    loss_mask = model.net.SigmoidCrossEntropyLoss(
        [blob_mask, 'masks_int32'],
        'loss_mask',
        scale=model.GetLossScale() * cfg.MRCNN.WEIGHT_LOSS_MASK
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_mask])
    model.AddLosses('loss_mask')
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Mask heads
# ---------------------------------------------------------------------------- #

def mask_rcnn_fcn_head_v1up4convs(model, blob_in, dim_in, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2."""
    return mask_rcnn_fcn_head_v1upXconvs(
        model, blob_in, dim_in, spatial_scale, 4
    )


def mask_rcnn_fcn_head_v1up4convs_gn(model, blob_in, dim_in, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2, with GroupNorm"""
    return mask_rcnn_fcn_head_v1upXconvs_gn(
        model, blob_in, dim_in, spatial_scale, 4
    )


def mask_rcnn_fcn_head_v1up(model, blob_in, dim_in, spatial_scale):
    """v1up design: 2 * (conv 3x3), convT 2x2."""
    return mask_rcnn_fcn_head_v1upXconvs(
        model, blob_in, dim_in, spatial_scale, 2
    )

def mask_rcnn_fcn_head_denseblock_4convs_1op(
    model, blob_in, dim_in, spatial_scale
):
    # Implemented fc fusion similar to PANet: https://arxiv.org/pdf/1803.01534.pdf
    # Implemented DenseBlock similar to DenseNet: https://arxiv.org/pdf/1611.09326.pdf
    """v1upXconvs design: X * (conv 3x3), convT 2x2."""
    ROI_Input = model.RoIFeatureTransform(
        blob_in,
        blob_out='_[mask]_roi_feat',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    dilation = cfg.MRCNN.DILATION
    dim_inner = cfg.MRCNN.DIM_REDUCED
    split_i = 0 # to keep track of i, may be redundant

      ## Printing out variables important for implementing fc fusion 
    logger = logging.getLogger(__name__)
    logger.info('Creating Custom Mask Branch...')
    logger.info('Implements Atrous Conv + DenseBlock + FC Fusion')

    ## Conv1 
    convfcn1 = model.Conv(
        ROI_Input,
        '_[mask]_fcn_1',
        dim_in,
        dim_inner,
        kernel=3,
        pad=1 * dilation,
        dilation=dilation,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    convfcn1 = model.Relu(convfcn1, convfcn1)

    conv_1_concat, _ = model.net.Concat(
        [ROI_Input,convfcn1],
        ['conv_1_concat_blob','conv_1_concat_blob_dim'],
        order = 'NCHW'
    )

    conv_1_1x1 = model.Conv(
        conv_1_concat,
        '_[mask]_conv_1_1x1',
        int(dim_in + dim_inner),
        dim_inner,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    conv_1_1x1 = model.Relu(conv_1_1x1, conv_1_1x1)

    ## Conv2
    convfcn2 = model.Conv(
        conv_1_1x1,
        '_[mask]_fcn_2',
        dim_inner,
        dim_inner,
        kernel=3,
        pad=1 * dilation,
        dilation=dilation,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    convfcn2 = model.Relu(convfcn2, convfcn2)

    ## Concat 2 block
    conv_2_concat, _ = model.net.Concat(
        [conv_1_concat,convfcn2],
        ['conv_2_concat_blob','conv_2_concat_blob_dim'],
        order = 'NCHW'
    )

    conv_2_1x1 = model.Conv(
        conv_2_concat,
        '_[mask]_conv_2_1x1',
        int(dim_in + 2 * dim_inner),
        dim_inner,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    conv_2_1x1 = model.Relu(conv_2_1x1, conv_2_1x1)

    ## Conv3
    convfcn3 = model.Conv(
        conv_2_1x1,
        '_[mask]_fcn_3',
        dim_inner,
        dim_inner,
        kernel=3,
        pad=1 * dilation,
        dilation=dilation,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    convfcn3 = model.Relu(convfcn3, convfcn3)

    ## Concat 3 block
    conv_3_concat, _ = model.net.Concat(
        [conv_2_concat,convfcn3],
        ['conv_3_concat_blob','conv_3_concat_blob_dim'],
        order = 'NCHW'
    )

    conv_3_1x1 = model.Conv(
        conv_3_concat,
        '_[mask]_conv_3_1x1',
        int(dim_in + 3 * dim_inner),
        dim_inner,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    conv_3_1x1 = model.Relu(conv_3_1x1, conv_3_1x1)

# Branch 1 - FCN
    ## Conv4
    convfcn4 = model.Conv(
        conv_3_1x1,
        '_[mask]_fcn_4',
        dim_inner,
        dim_inner,
        kernel=3,
        pad=1,
        # dilation=dilation,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    convfcn4 = model.Relu(convfcn4, convfcn4)

    ## Concat 4 block
    conv_4_concat, _ = model.net.Concat(
        [convfcn1, convfcn2, convfcn3, convfcn4],
        ['conv_4_concat_blob','conv_4_concat_blob_dim'],
        order = 'NCHW'
    )

# Implementing FC Fusion
# Splitting into 2 branches here
# First branch consists of FCN, the second branch as a fc layer along with FCN

# Branch 1 - FCN
# Man! this input is big AF!, try to optimize it
    convfcn5 = model.Conv(
        conv_4_concat,
        '_[mask]_fcn_5',
        int(dim_inner * 4), # 256x4 !!
        dim_inner,
        kernel=3, # 3!
        pad=1,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    convfcn5 = model.Relu(convfcn5, convfcn5)

    # Upsample layer
    model.ConvTranspose(
        convfcn5,
        'conv5_mask_fcn',
        dim_inner,
        dim_inner,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask_fcn = model.Relu('conv5_mask_fcn', 'conv5_mask_fcn')

# Branch 2 - fc + FCN

# To reduce channel dimensions
    conv_4_1x1_fc = model.Conv(
        conv_4_concat,
        '_[mask]_fc_4_1x1',
        int(dim_inner * 4),
        dim_inner,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    conv_4_1x1_fc = model.Relu(conv_4_1x1_fc, conv_4_1x1_fc)

    convfc1 = model.Conv(
        conv_4_1x1_fc,
        '_[mask]_fc_1',
        dim_inner,
        dim_inner,
        kernel=3,
        pad=1,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    convfc1 = model.Relu(convfc1, convfc1)

    # Conv layer to reduce no. of channels to reduce computation
    convfc2 = model.Conv(
        convfc1,
        '_[mask]_fc_2',
        dim_inner,
        int(dim_inner/2),
        kernel=3,
        pad=1,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    convfc2 = model.Relu(convfc2, convfc2)

    # fc layer
    convfc3 = model.FC(
            convfc2,
            '_[mask]_fc_3',
            int(dim_inner/2) * cfg.MRCNN.ROI_XFORM_RESOLUTION**2, # 128*14*14
            4 * cfg.MRCNN.ROI_XFORM_RESOLUTION**2, # 4*14*14 = 28*28
            weight_init=gauss_fill(0.001),
            bias_init=const_fill(0.0)
    )


    # Reshape fc layer to add to FCN layer of the other branch
    # Note that this shape is different from the final FCN layer of the other branch
    model.net.Reshape(
        ['_[mask]_fc_3'], # [Input]
        ['_[mask]_fc_reshaped', '_[mask]_fc_old_shaped_3'], # [Output, old_shape]
        shape=(-1,1,cfg.MRCNN.ROI_XFORM_RESOLUTION*2,cfg.MRCNN.ROI_XFORM_RESOLUTION*2) # shape = (n,c,h,w)
    )

    # Reshape with 1x1 conv to match shape of the final FCN layer of the other branch
    # This next step is not recommended, change it when you get a better idea in order to save memory.
    # TODO: Freeze this layer
    convfc_mask = model.Conv(
        '_[mask]_fc_reshaped',
        '_[mask]_fc_bg_fg',
        1,
        dim_inner,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=const_fill(1.0),
        bias_init=const_fill(0.0)
    )
    blob_mask_fc = model.Relu('_[mask]_fc_bg_fg', '_[mask]_fc_bg_fg')

    # Adding the 2 branches
    blob_mask = model.net.Sum([blob_mask_fcn, blob_mask_fc],'fc_fusion_mask')

    return blob_mask, dim_inner

def mask_rcnn_fcn_head_v1upXconvs(
    model, blob_in, dim_in, spatial_scale, num_convs
):
    # Implemented fc fusion similar to PANet: https://arxiv.org/pdf/1803.01534.pdf
    # TODO: Modify config file to include option to implement fc_fusion
    # TODO: Add fc_fusion layers in a if condition.

    """v1upXconvs design: X * (conv 3x3), convT 2x2."""
    current = model.RoIFeatureTransform(
        blob_in,
        blob_out='_[mask]_roi_feat',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    dilation = cfg.MRCNN.DILATION
    dim_inner = cfg.MRCNN.DIM_REDUCED
    split_i = 0 # to keep track of i, may be redundant

      ## Printing out variables important for implementing fc fusion 
    logger = logging.getLogger(__name__)
    logger.info('Creating Mask Branch: mask_rcnn_fcn_head_v1upXconvs ....')
    logger.info('Implements FC Fusion')
    # logger.info(dim_inner/2 * cfg.MRCNN.RESOLUTION * cfg.MRCNN.RESOLUTION ) ## 100352 = 128*28*28
    # logger.info(dim_inner/2 ) ## 128
    # logger.info(cfg.MRCNN.DIM_REDUCED) ## 256
    # logger.info(cfg.MODEL.NUM_CLASSES) ## 81
    # logger.info(cfg.MODEL.NUM_CLASSES if cfg.MRCNN.CLS_SPECIFIC_MASK else 1) ## 81
    # logger.info(cfg.MRCNN.CLS_SPECIFIC_MASK) ## True
    # logger.info(cfg.MRCNN.RESOLUTION) ## 28 (by default, it is 14, but changed to 28)
    # logger.info(cfg.MRCNN.ROI_XFORM_METHOD) ## ROIAlign
    # logger.info(cfg.MRCNN.ROI_XFORM_RESOLUTION) ## 14 -> Use this for H,W in (N,C,H,W)
    # logger.info(cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO) ## 2

    # Split branches from penultimate layer
    for i in range(num_convs - 1):
        current = model.Conv(
            current,
            '_[mask]_fcn' + str(i + 1),
            dim_in,
            dim_inner,
            kernel=3,
            dilation=dilation,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = dim_inner
        split_i = i + 1

# Implementing FC Fusion
# Splitting into 2 branches
# First branch consists of FCN, the second branch as a fc layer along with FCN

# Branch 1 - FCN
# TODO: no dilation in branches
    convfcn1 = model.Conv(
        current,
        '_[mask]_fcn' + str(split_i + 1),
        dim_in,
        dim_inner,
        kernel=3,
        pad=1,
        # dilation=dilation,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    convfcn1_r = model.Relu(convfcn1, convfcn1)

    # Upsample layer
    model.ConvTranspose(
        convfcn1_r,
        'conv5_mask_fcn',
        dim_inner,
        dim_inner,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask_fcn = model.Relu('conv5_mask_fcn', 'conv5_mask_fcn')

# Branch 2 - fc + FCN

    convfc1 = model.Conv(
        current,
        '_[mask]_fc' + str(split_i + 1),
        dim_inner,
        dim_inner,
        kernel=3,
        pad=1 * dilation,
        dilation=dilation,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    convfc1= model.Relu(convfc1, convfc1)

    # Conv layer to reduce no. of channels to reduce computation
    convfc2 = model.Conv(
        convfc1,
        '_[mask]_fc' + str(split_i + 2),
        dim_inner,
        int(dim_inner/2),
        kernel=3,
        pad=1,
        #dilation=dilation,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    convfc2_r = model.Relu(convfc2, convfc2)

    # fc layer
    convfc3 = model.FC(
            convfc2_r,
            '_[mask]_fc' + str(split_i + 3),
            int(dim_inner/2) * cfg.MRCNN.ROI_XFORM_RESOLUTION**2, # 128*14*14
            4 * cfg.MRCNN.ROI_XFORM_RESOLUTION**2, # 4*14*14 = 28*28
            weight_init=gauss_fill(0.001),
            bias_init=const_fill(0.0)
    )


    # Reshape fc layer to add to FCN layer of the other branch
    # Note that this shape is different from the final FCN layer of the other branch
    model.net.Reshape(
        ['_[mask]_fc' + str(split_i + 3)], # [Input]
        ['_[mask]_fc_reshaped', '_[mask]_fc_old_shaped' + str(split_i + 3)], # [Output, old_shape]
        shape=(-1,1,cfg.MRCNN.ROI_XFORM_RESOLUTION*2,cfg.MRCNN.ROI_XFORM_RESOLUTION*2) # shape = (n,c,h,w)
    )

    # Reshape with 1x1 conv to match shape of the final FCN layer of the other branch
    # This next step is not recommended, change it when you get a better idea in order to save memory.
    # TODO: Freeze this layer
    convfc_mask = model.Conv(
        '_[mask]_fc_reshaped',
        '_[mask]_fc_bg_fg',
        1,
        dim_inner,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=const_fill(1.0),
        bias_init=const_fill(0.0)
    )
    blob_mask_fc = model.Relu('_[mask]_fc_bg_fg', '_[mask]_fc_bg_fg')

    # Adding the 2 branches
    blob_mask = model.net.Sum([blob_mask_fcn, blob_mask_fc],'fc_fusion_mask')

    return blob_mask, dim_inner


def mask_rcnn_fcn_head_v1upXconvs_gn(
    model, blob_in, dim_in, spatial_scale, num_convs
):
    """v1upXconvs design: X * (conv 3x3), convT 2x2, with GroupNorm"""
    current = model.RoIFeatureTransform(
        blob_in,
        blob_out='_mask_roi_feat',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    dilation = cfg.MRCNN.DILATION
    dim_inner = cfg.MRCNN.DIM_REDUCED
    split_i = 0 # to keep track of i

    for i in range(num_convs - 1): # default-> range(num_convs)
        # branches out from one layer before the last layer
        current = model.ConvGN(
            current,
            '_[mask]_fcn' + str(i + 1),
            dim_in,
            dim_inner,
            group_gn=get_group_gn(dim_inner),
            kernel=3,
            dilation=dilation,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = dim_inner
        split_i = i + 1
# Splitting into branches

# Branch 1 - FCN
    convfcn1 = model.ConvGN(
        current,
        '_[mask]_fcn' + str(split_i + 1),
        dim_inner,
        dim_inner,
        group_gn=get_group_gn(dim_inner),
        kernel=3,
        pad=1 * dilation,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    convfcn1_r = model.Relu(convfcn1, convfcn1)

    # upsample layer
    model.ConvTranspose(
        convfcn1_r,
        'conv5_mask_fcn',
        dim_inner,
        dim_inner,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask_fcn = model.Relu('conv5_mask_fcn', 'conv5_mask_fcn')

# Branch 2 - fc + FCN
    convfc1 = model.ConvGN(
        current,
        '_[mask]_fc' + str(split_i + 1),
        dim_inner,
        dim_inner,
        group_gn=get_group_gn(dim_inner),
        kernel=3,
        pad=1 * dilation,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    convfc1_r = model.Relu(convfc1, convfc1)

    # Conv layer to reduce no. of channels to reduce computation
    convfc2 = model.ConvGN(
        convfc1_r,
        '_[mask]_fc' + str(split_i + 2),
        dim_inner,
        int(dim_inner / 2),
        group_gn=get_group_gn(int(dim_inner / 2)),
        kernel=3,
        pad=1 * dilation,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    convfc2_r = model.Relu(convfc2, convfc2)

    # fc layer
    convfc3 = model.FC(
            convfc2_r,
            '_[mask]_fc' + str(split_i + 3),
            int(dim_inner/2) * cfg.MRCNN.ROI_XFORM_RESOLUTION**2, # 128*14*14
            4 * cfg.MRCNN.ROI_XFORM_RESOLUTION**2, # 4*14*14 = 28*28
            weight_init=gauss_fill(0.001),
            bias_init=const_fill(0.0)
    )

    # Intentional error to stop code and read values in log
    #model.net.Reshape(3,a)

    # Reshape fc layer to add to FCN layer of the other branch
    # Note that this shape is different from the final FCN layer of the other branch

    model.net.Reshape(
        ['_[mask]_fc' + str(split_i + 3)], # [Input]
        ['_[mask]_fc_reshaped', '_[mask]_fc_old_shaped' + str(split_i + 3)], # [Output, old_shape]
        shape=(-1,1,cfg.MRCNN.ROI_XFORM_RESOLUTION*2,cfg.MRCNN.ROI_XFORM_RESOLUTION*2) # shape = (n,c,h,w)
    )

    # Reshape with 1x1 conv to match shape of the final FCN layer of the other branch
    # This next step is not recommended, change it when you get a better idea in order to save memory.
    # TODO: Freeze this layer
    convfc_mask = model.Conv(
        '_[mask]_fc_reshaped',
        '_[mask]_fc_bg_fg',
        1,
        dim_inner,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=const_fill(1.0),
        bias_init=const_fill(0.0)
    )
    blob_mask_fc = model.Relu('_[mask]_fc_bg_fg', '_[mask]_fc_bg_fg')

    # Adding the 2 branches
    blob_mask = model.net.Sum([blob_mask_fcn, blob_mask_fc],'fc_fusion_mask')

    return blob_mask, dim_inner


def mask_rcnn_fcn_head_v0upshare(model, blob_in, dim_in, spatial_scale):
    """Use a ResNet "conv5" / "stage5" head for mask prediction. Weights and
    computation are shared with the conv5 box head. Computation can only be
    shared during training, since inference is cascaded.

    v0upshare design: conv5, convT 2x2.
    """
    # Since box and mask head are shared, these must match
    assert cfg.MRCNN.ROI_XFORM_RESOLUTION == cfg.FAST_RCNN.ROI_XFORM_RESOLUTION

    if model.train:  # share computation with bbox head at training time
        dim_conv5 = 2048
        blob_conv5 = model.net.SampleAs(
            ['res5_2_sum', 'roi_has_mask_int32'],
            ['_[mask]_res5_2_sum_sliced']
        )
    else:  # re-compute at test time
        blob_conv5, dim_conv5 = add_ResNet_roi_conv5_head_for_masks(
            model,
            blob_in,
            dim_in,
            spatial_scale
        )

    dim_reduced = cfg.MRCNN.DIM_REDUCED

    blob_mask = model.ConvTranspose(
        blob_conv5,
        'conv5_mask',
        dim_conv5,
        dim_reduced,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),  # std only for gauss
        bias_init=const_fill(0.0)
    )
    model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_reduced


def mask_rcnn_fcn_head_v0up(model, blob_in, dim_in, spatial_scale):
    """v0up design: conv5, deconv 2x2 (no weight sharing with the box head)."""
    blob_conv5, dim_conv5 = add_ResNet_roi_conv5_head_for_masks(
        model,
        blob_in,
        dim_in,
        spatial_scale
    )

    dim_reduced = cfg.MRCNN.DIM_REDUCED

    model.ConvTranspose(
        blob_conv5,
        'conv5_mask',
        dim_conv5,
        dim_reduced,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=('GaussianFill', {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_reduced


def add_ResNet_roi_conv5_head_for_masks(model, blob_in, dim_in, spatial_scale):
    """Add a ResNet "conv5" / "stage5" head for predicting masks."""
    model.RoIFeatureTransform(
        blob_in,
        blob_out='_[mask]_pool5',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    dilation = cfg.MRCNN.DILATION
    stride_init = int(cfg.MRCNN.ROI_XFORM_RESOLUTION / 7)  # by default: 2

    s, dim_in = ResNet.add_stage(
        model,
        '_[mask]_res5',
        '_[mask]_pool5',
        3,
        dim_in,
        2048,
        512,
        dilation,
        stride_init=stride_init
    )

    return s, 2048

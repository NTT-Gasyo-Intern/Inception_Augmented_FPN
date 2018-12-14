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


def mask_rcnn_fcn_head_v1upXconvs(
    model, blob_in, dim_in, spatial_scale, num_convs
):
    # Implemented fc fusion similar to PANet: https://arxiv.org/pdf/1803.01534.pdf
    # TODO: Modify config file to include option to implement fc_fusion
    # TODO: Add fc_fusion layers in a if condition.

    """v1upXconvs design: X * (conv 3x3), convT 2x2."""
    input_blob = model.RoIFeatureTransform(
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
    hidden_dim_fc = cfg.FAST_RCNN.MLP_HEAD_DIM

      ## Printing out variables important for implementing fc fusion 
    logger = logging.getLogger(__name__)
    logger.info('FC Fusion Rev 4.1')
    logger.info('Implementing -> SENet + DenseNet + FC_Fusion')
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
    # logger.info(hidden_dim_fc) ## 2
    # logger.info(4 * cfg.MRCNN.ROI_XFORM_RESOLUTION**2) ## 784
    # logger.info('******************************')
    # logger.info(cfg.FAST_RCNN.ROI_XFORM_RESOLUTION) ## 7

    #
    # Global Attention
    #

    global_in = model.AveragePool(
        input_blob,
        '_[mask]_global_in',
        global_pooling=True
    )

    model.net.Reshape(
        [global_in], # [Input]
        ['_[mask]_global_in_reshaped', '_[mask]_global_in_old_shape'], # [Output, old_shape]
        shape=(-1,dim_in,1,1) # shape = (n,c,h,w)
    )

    global_fc_1 = model.FC(
        '_[mask]_global_in_reshaped',
        '_[mask]_global_fc_1',
        dim_in,
        int(dim_in / 4),
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    global_fc_1 = model.Relu(global_fc_1, global_fc_1)

    global_fc_2 = model.FC(
        '_[mask]_global_fc_1',
        '_[mask]_global_fc_2',
        int(dim_in / 4),
        dim_in,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    global_fc_2 = model.Sigmoid(global_fc_2, global_fc_2)

    model.net.Reshape(
        [global_fc_2], # [Input]
        ['_[mask]_global_fc_2_reshaped', '_[mask]_global_fc_2_old_shape'], # [Output, old_shape]
        shape=(-1,dim_in,1,1) # shape = (n,c,h,w)
    )

    global_scaled = model.net.UpsampleNearest(
        '_[mask]_global_fc_2_reshaped',
        '_[mask]_global_scaled',
        scale=cfg.MRCNN.ROI_XFORM_RESOLUTION
    )

    current = model.net.Sum(
        ['_[mask]_global_scaled', input_blob],
        '_[mask]_global_out'
    )

    #
    # End of Global Attention
    #

    # add_mask_inception_module(
    #     model,
    #     current,                                      # input blob
    #     '_[mask]_inception_out_0',                                     # output blob
    #     dim_in,        # input dimension
    #     dim_inner,                                    # output dimension
    #     0 # Id number(if multiple blocks are used: use 0,1,2 ...)
    # )

    # add_mask_inception_module(
    #     model,
    #     '_[mask]_inception_out_0',                                      # input blob
    #     '_[mask]_inception_out_1',                                     # output blob
    #     dim_inner,        # input dimension
    #     dim_inner,                                    # output dimension
    #     1 # Id number(if multiple blocks are used: use 0,1,2 ...)
    # )

    # main_branch = model.net.Sum(
    #     ['[mask]_inception_out_0', '[mask]_inception_out_1'],
    #     '[mask]_inception_sum'
    # )

    main_branch = model.Conv(
        current,
        '_[mask]_main_br_0', 
        dim_in,
        dim_inner,
        kernel=3,
        dilation=dilation,
        pad=1 * dilation,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    main_branch = model.Relu(main_branch, main_branch)

    main_branch = model.Conv(
        '_[mask]_main_br_0',
        '_[mask]_main_br_1', 
        dim_inner,
        dim_inner,
        kernel=3,
        dilation=dilation,
        pad=1 * dilation,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    main_branch = model.Relu(main_branch, main_branch)

    main_br_1_concat, _ = model.net.Concat(
        ['_[mask]_main_br_1',input_blob],
        ['_[mask]_main_br_1_concat','_[mask]_main_br_1_concat_dim'],
        order = 'NCHW'
    )

    main_branch = model.Conv(
        '_[mask]_main_br_1_concat',
        '_[mask]_main_br_1_concat_1x1',
        dim_in=int(dim_inner + dim_in),
        dim_out=dim_inner,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    main_branch = model.Relu(main_branch, main_branch)    

    for i in range(num_convs - 3):
        main_branch = model.Conv(
            main_branch,
            '_[mask]_fcn_main' + str(i + 2),
            dim_inner,
            dim_inner,
            kernel=3,
            dilation=dilation,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        main_branch = model.Relu(main_branch, main_branch)

    main_branch_sum = model.net.Sum(
        ['_[mask]_main_br_0', main_branch],
        '_[mask]_main_br_sum'
    )
    main_branch_sum = model.Relu(main_branch_sum, main_branch_sum)

# Implementing FC Fusion ver 2
# Splitting into 2 branches
# First branch consists of FCN, the second branch as a fc layer along with FCN

# Branch 1 - FCN

    convfcn1 = model.Conv(
        main_branch_sum,
        '_[mask]_fcn_0',
        dim_inner,
        dim_inner,
        kernel=3,
        pad=1 * dilation,
        dilation=dilation,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    convfcn1_r = model.Relu(convfcn1, convfcn1)

    convfcn1_r_sum_1 = model.net.Sum(
        ['_[mask]_main_br_1_concat_1x1', convfcn1_r],
        '_[mask]_fcn_0_sum_1'
    )
    convfcn1_r_sum_1 = model.Relu(convfcn1_r_sum_1, convfcn1_r_sum_1)

    convfcn1_r_sum_2 = model.net.Sum(
        ['_[mask]_main_br_0', '_[mask]_fcn_0_sum_1'],
        '_[mask]_fcn_0_sum_2'
    )

    #
    # Global Attention 2
    #

    fcn_global_in = model.AveragePool(
        convfcn1_r_sum_2,
        '_[mask]_fcn_global_in',
        global_pooling=True
    )

    model.net.Reshape(
        [fcn_global_in], # [Input]
        ['_[mask]_fcn_global_in_reshaped', '_[mask]_fcn_global_in_old_shape'], # [Output, old_shape]
        shape=(-1,dim_in,1,1) # shape = (n,c,h,w)
    )

    fcn_global_fc_1 = model.FC(
        '_[mask]_fcn_global_in_reshaped',
        '_[mask]_fcn_global_fc_1',
        dim_inner,
        int(dim_inner / 4),
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    fcn_global_fc_1 = model.Relu(fcn_global_fc_1, fcn_global_fc_1)

    fcn_global_fc_2 = model.FC(
        '_[mask]_fcn_global_fc_1',
        '_[mask]_fcn_global_fc_2',
        int(dim_inner / 4),
        dim_inner,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    fcn_global_fc_2 = model.Sigmoid(fcn_global_fc_2, fcn_global_fc_2)

    model.net.Reshape(
        [fcn_global_fc_2], # [Input]
        ['_[mask]_fcn_global_fc_2_reshaped', '_[mask]_fcn_global_fc_2_old_shape'], # [Output, old_shape]
        shape=(-1,dim_in,1,1) # shape = (n,c,h,w)
    )

    fcn_global_scaled = model.net.UpsampleNearest(
        '_[mask]_fcn_global_fc_2_reshaped',
        '_[mask]_fcn_global_scaled',
        scale=cfg.MRCNN.ROI_XFORM_RESOLUTION
    )

    fcn_global_out = model.net.Sum(
        ['_[mask]_fcn_global_scaled', convfcn1_r_sum_2],
        '_[mask]_fcn_global_out'
    )

    #
    # End of Global Attention 2
    #

    # Upsample layer
    model.ConvTranspose(
        fcn_global_out,
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
        main_branch_sum,
        '_[mask]_fc_0',
        dim_inner,
        dim_inner,
        kernel=3,
        pad=1 * dilation,
        dilation=dilation,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    convfc1_r = model.Relu(convfc1, convfc1)

    # deconv to increase spatial dim and also reduce channels -> decrease comp
    # convfc2 = model.Conv(
    #     convfc1_r,
    #     '_[mask]_fc_1',
    #     dim_inner,
    #     int(dim_inner/2),
    #     kernel=3,
    #     pad=1 * dilation,
    #     dilation=dilation,
    #     stride=1,
    #     weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
    #     bias_init=const_fill(0.0)
    # )
    # convfc2_r = model.Relu(convfc2, convfc2)

    convfc2 = model.ConvTranspose(
        convfc1_r,
        '_[mask]_fc_1',
        dim_inner,
        int(dim_inner/8),
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    convfc2_r = model.Relu('_[mask]_fc_1', '_[mask]_fc_1')

    # Now this dimension is 28*28*32 i,e (2*cfg.MRCNN.ROI_XFORM_RESOLUTION)^2 * dim_inner/8

    #
    # Capillary from bb branch
    #

    # convfc2_fc6 = model.FC(
    #     'fc6',
    #     '_[mask]_fc_1_fc6',
    #     hidden_dim_fc, # 1024
    #     4 * cfg.MRCNN.ROI_XFORM_RESOLUTION**2, # 28*28*1 = 784
    #     weight_init=gauss_fill(0.001),
    #     bias_init=const_fill(0.0)
    # )
    # convfc2_fc6 = model.Relu(convfc2_fc6, convfc2_fc6)

    # model.net.Reshape(
    #     ['_[mask]_fc_1_fc6'], # [Input]
    #     ['_[mask]_fc_1_fc6_reshaped', 'fc6_old_shape'], # [Output, old_shape]
    #     shape=(-1,1,cfg.MRCNN.ROI_XFORM_RESOLUTION*2,cfg.MRCNN.ROI_XFORM_RESOLUTION*2) # shape = (n,c,h,w)
    # )

    #
    # End of capillary
    #

    # convfc3, _ = model.net.Concat(
    #     ['_[mask]_fc_1_fc6_reshaped',convfc2_r],
    #     ['_[mask]_fc_2','_[mask]_fc_1_dim'],
    #     order = 'NCHW'
    # )

    # main fc layer
    convfc4 = model.FC(
        # convfc3,
        # '_[mask]_fc_3',
        # int(dim_inner/8 * 4 * cfg.MRCNN.ROI_XFORM_RESOLUTION**2) + 1, # 32*4*14*14+1 = 32*28*28+1
        convfc2_r,
        '_[mask]_fc_3',
        int(dim_inner/8 * 4 * cfg.MRCNN.ROI_XFORM_RESOLUTION**2), # 32*4*14*14+1 = 32*28*28+1
        4 * cfg.MRCNN.ROI_XFORM_RESOLUTION**2, # 4*14*14 = 28*28
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    convfc4 = model.Relu(convfc4, convfc4)

    # Reshape fc layer to add to FCN layer of the other branch
    # Note that this shape is different from the final FCN layer of the other branch
    model.net.Reshape(
        [convfc4], # [Input]
        ['_[mask]_convfc4_reshaped', '_[mask]_convfc4_reshaped_old_shape'], # [Output, old_shape]
        shape=(-1,1,cfg.MRCNN.ROI_XFORM_RESOLUTION*2,cfg.MRCNN.ROI_XFORM_RESOLUTION*2) # shape = (n,c,h,w)
    )

    #
    # Branch combination
    #

    branch_combo, _ = model.net.Concat(
        ['_[mask]_convfc4_reshaped',blob_mask_fcn],
        ['_[mask]_branch_combo','_[mask]_branch_combo_dim'],
        order = 'NCHW'
    )

    blob_mask = model.Conv(
        branch_combo,
        'fc_fusion_mask',
        dim_in=int(dim_inner + 1),
        dim_out=dim_inner,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask = model.Relu(blob_mask, blob_mask)

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

def add_mask_inception_module( 
    model, inception_input, inception_output, dim_in, dim_out, num = 0
):
    ##### 1x1 path or l_path #####
    if cfg.FPN.USE_GN:
        # use GroupNorm
        l_path = model.ConvGN(
            inception_input,
            '_[mask]_l_path_blob_' + str(num),
            dim_in=dim_in,
            dim_out=int(dim_out / 4),
            group_gn=get_group_gn(int(dim_out / 4)),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=('XavierFill', {}),
            bias_init=const_fill(0.0)
        )
    else:
        l_path = model.Conv(
            inception_input,
            '_[mask]_l_path_blob_' + str(num),
            dim_in=dim_in,
            dim_out=int(dim_out / 4),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=('XavierFill', {}),
            bias_init=const_fill(0.0)
        )
    l_path_op = model.Relu(l_path, '_[mask]_l_path_blob_op_' + str(num) + '_relu')

    ##### 3x3 path or m_path #####
    # 1x1
    if cfg.FPN.USE_GN:
        # use GroupNorm
        m_path_1 = model.ConvGN(
            inception_input,
            '_[mask]_m_path_blob_1_' + str(num),
            dim_in=dim_in,
            dim_out=int(dim_out / 2),
            group_gn=get_group_gn(int(dim_out / 2)),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=('XavierFill', {}),
            bias_init=const_fill(0.0)
        )
    else:
        m_path_1 = model.Conv(
            inception_input,
            '_[mask]_m_path_blob_1_' + str(num),
            dim_in=dim_in,
            dim_out=int(dim_out / 2),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=('XavierFill', {}),
            bias_init=const_fill(0.0)
        )
    m_path_1 = model.Relu(m_path_1, '_[mask]_m_path_blob_1_' + str(num) + '_relu')

    # 3x3
    if cfg.FPN.USE_GN:
        # use GroupNorm
        m_path_2 = model.ConvGN(
            m_path_1,
            '_[mask]_m_path_blob_2_' + str(num),
            dim_in=int(dim_out / 2),
            dim_out=int(dim_out / 4),
            group_gn=get_group_gn(int(dim_out / 4)),
            kernel=3,
            pad=1,
            stride=1,
            weight_init=('XavierFill', {}),
            bias_init=const_fill(0.0)
        )
    else:
        m_path_2 = model.Conv(
            m_path_1,
            '_[mask]_m_path_blob_2_' + str(num),
            dim_in=int(dim_out / 2),
            dim_out=int(dim_out / 4),
            kernel=3,
            pad=1,
            stride=1,
            weight_init=('XavierFill', {}),
            bias_init=const_fill(0.0)
        )
    m_path_op = model.Relu(m_path_2, '_[mask]_m_path_blob_op_' + str(num) + '_relu')

    ##### 5x5 = 2x(3x3) path or s_path #####
    # 1x1 - part1
    if cfg.FPN.USE_GN:
        # use GroupNorm
        s_path_1 = model.ConvGN(
            inception_input,
            '_[mask]_s_path_blob_1_' + str(num),
            dim_in=dim_in,
            dim_out=int(dim_out / 4),
            group_gn=get_group_gn(int(dim_out / 2)),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=('XavierFill', {}),
            bias_init=const_fill(0.0)
        )
    else:
        s_path_1 = model.Conv(
            inception_input,
            '_[mask]_s_path_blob_1_' + str(num),
            dim_in=dim_in,
            dim_out=int(dim_out / 2),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=('XavierFill', {}),
            bias_init=const_fill(0.0)
        )
    s_path_1 = model.Relu(s_path_1, '_[mask]_s_path_blob_1_' + str(num) + '_relu')

    # 3x3 - part2
    if cfg.FPN.USE_GN:
        # use GroupNorm
        s_path_2 = model.ConvGN(
            s_path_1,
            '_[mask]_s_path_blob_2_' + str(num),
            dim_in=int(dim_out / 2),
            dim_out=int(dim_out / 2),
            group_gn=get_group_gn(dim_out / 2),
            kernel=3,
            pad=1,
            stride=1,
            weight_init=('XavierFill', {}),
            bias_init=const_fill(0.0)
        )
    else:
        s_path_2 = model.Conv(
            s_path_1,
            '_[mask]_s_path_blob_2_' + str(num),
            dim_in=int(dim_out / 2),
            dim_out=int(dim_out / 2),
            kernel=3,
            pad=1,
            stride=1,
            weight_init=('XavierFill', {}),
            bias_init=const_fill(0.0)
        )
    s_path_2 = model.Relu(s_path_2, '_[mask]_s_path_blob_2_' + str(num) + '_relu')

    # 3x3 - part3
    if cfg.FPN.USE_GN:
        # use GroupNorm
        s_path_3 = model.ConvGN(
            s_path_2,
            '_[mask]_s_path_blob_3_' + str(num),
            dim_in=int(dim_out / 2),
            dim_out=int(dim_out / 4),
            group_gn=get_group_gn(dim_out / 4),
            kernel=3,
            pad=1,
            stride=1,
            weight_init=('XavierFill', {}),
            bias_init=const_fill(0.0)
        )
    else:
        s_path_3 = model.Conv(
            s_path_2,
            '_[mask]_s_path_blob_3_' + str(num),
            dim_in=int(dim_out / 2),
            dim_out=int(dim_out / 4),
            kernel=3,
            pad=1,
            stride=1,
            weight_init=('XavierFill', {}),
            bias_init=const_fill(0.0)
        )
    s_path_op = model.Relu(s_path_3, '_[mask]_s_path_blob_op_' + str(num) + '_relu')

    ##### MaxPool path or xs_path #####
    # MaxPool - part1
    xs_path_1 = model.MaxPool(
        inception_input,
        '_[mask]_xs_path_blob_1_' + str(num),
        kernel=3,
        pad=1,
        stride=1
    )

    # 1x1 - part2
    if cfg.FPN.USE_GN:
        # use GroupNorm
        xs_path_2 = model.ConvGN(
            xs_path_1,
            '_[mask]_xs_path_blob_2_' + str(num),
            dim_in=dim_in,
            dim_out=int(dim_out / 4),
            group_gn=get_group_gn(int(dim_out / 4)),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=('XavierFill', {}),
            bias_init=const_fill(0.0)
        )
    else:
        xs_path_2 = model.Conv(
            xs_path_1,
            '_[mask]_xs_path_blob_2_' + str(num),
            dim_in=dim_in,
            dim_out=int(dim_out / 4),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=('XavierFill', {}),
            bias_init=const_fill(0.0)
        )
    xs_path_op = model.Relu(xs_path_2, '_[mask]_xs_path_blob_2_' + str(num) + '_relu')

    ##### Concat #####
    model.net.Concat(
        [l_path_op, m_path_op, s_path_op, xs_path_op],
        [inception_output,'_[mask]_inception_output_fpn_' + str(num) + '_dim'],
        order = 'NCHW'
    )

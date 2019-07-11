# Densely Connected FPN

## Instructions for running the model:

- Download and install Detectron (based on Caffe2).
  https://github.com/facebookresearch/Detectron
  
- Replace the folder /configs in detectron with the configs folder provided

- Replace the folder detectron/modeling with the modeling folder provided

- Run the model with --cfg parameter set as e2e_mask_rcnn_R-50-FPN_1x.yaml. Refer the link below for more instructions.
  https://github.com/facebookresearch/Detectron/blob/master/GETTING_STARTED.md

## Description of folders:

- MaskRCNN-visualizations/vis -> Visualization of Mask RCNN (original)
- conf_1_20181018--InceptionCCAdd -> Model 2
- conf_2_20181019--InceptionAdd -> Model 3
- conf_3_20181026--InceptionCCOnly -> Model 1*
- conf_4_20181029--InceptionCC_ph -> Model 1
- journal_1_20181214--DMB_Inc -> Model 1 + Dense Mask Branch
- journal_2_20181214--DMB -> Mask RCNN + Dense Mask Branch

### Note: The /vis folder inside the above folders contain the visualization of the respective models.

import tensorflow as tf

from tensorflow.keras import layers
from keras_cv_attention_models import efficientnet
from keras_cv_attention_models import swin_transformer_v2 as swin_v2
from layers.biFPN import bi_fpn, det_header_pre, det_header_post
from layers.fpn import build_FPN
from layers.maskNet import MaskHead
from tensorflow.keras import initializers
from layers.head import PredictionModule, FastMaskIoUNet
assert tf.__version__.startswith('2')
from detection import Detect
from data import anchor
import numpy as np

class MaskED(tf.keras.Model):
    """
        Creating the MaskED Architecture
        Arguments:

    """

    def __init__(self, config):
        super(MaskED, self).__init__()

        backbones = {
                    "efficientdet_b0": efficientnet.EfficientNetV1B0,
                    "efficientdet_b1": efficientnet.EfficientNetV1B1,
                    "efficientdet_b2": efficientnet.EfficientNetV1B2,
                    "efficientdet_b3": efficientnet.EfficientNetV1B3,
                    "efficientdet_b4": efficientnet.EfficientNetV1B4,
                    "efficientdet_b5": efficientnet.EfficientNetV1B5,
                    "efficientdet_b6": efficientnet.EfficientNetV1B6,
                    "efficientdet_b7": efficientnet.EfficientNetV1B6,
                    "efficientdet_b7x": efficientnet.EfficientNetV1B7,
                    "efficientdet_lite0": efficientnet.EfficientNetV1Lite0,
                    "efficientdet_lite1": efficientnet.EfficientNetV1Lite1,
                    "efficientdet_lite2": efficientnet.EfficientNetV1Lite2,
                    "efficientdet_lite3": efficientnet.EfficientNetV1Lite3,
                    "efficientdet_lite3x": efficientnet.EfficientNetV1Lite3,
                    "efficientdet_lite4": efficientnet.EfficientNetV1Lite4,
                    'swin-tiny' : swin_v2.SwinTransformerV2Tiny_window8,
                    }
        out_layers = {'efficientnetv2b0': [
                                            "stack_2_block1_output", 
                                            "stack_4_block2_output", 
                                            "stack_6_block0_output"],
                      'efficientnetv2b1': [
                                            "stack_2_block2_output", 
                                            "stack_4_block3_output", 
                                            "stack_6_block1_output"],
                      'efficientnetv2b2': [
                                            "stack_2_block2_output", 
                                            "stack_4_block3_output", 
                                            "stack_6_block1_output"],
                      'efficientnetv2b3': [
                                            "stack_2_block2_output", 
                                            "stack_4_block4_output", 
                                            "stack_6_block1_output"],
                      'efficientnetv2b4': [
                                            "stack_2_block3_output", 
                                            "stack_4_block5_output", 
                                            "stack_6_block1_output"],
                      'efficientnetv2b5': [
                                            "stack_2_block4_output", 
                                            "stack_4_block6_output", 
                                            "stack_6_block2_output"],
                      'efficientnetv2b6': [
                                            "stack_2_block5_output", 
                                            "stack_4_block7_output", 
                                            "stack_6_block2_output"],
                      'efficientnetv2b7': [
                                            "stack_2_block5_output", 
                                            "stack_4_block7_output", 
                                            "stack_6_block2_output"],
                      'efficientnetv2b7x': [
                                            "stack_2_block6_output", 
                                            "stack_4_block9_output", 
                                            "stack_6_block3_output"]
                      'efficientnetlite0': [
                                            "stack_2_block1_output", 
                                            "stack_4_block2_output", 
                                            "stack_6_block0_output"],
                      'efficientnetlite1': [
                                            "stack_2_block2_output", 
                                            "stack_4_block3_output", 
                                            "stack_6_block0_output"],
                      'efficientnetlite2': [
                                            "stack_2_block2_output", 
                                            "stack_4_block3_output", 
                                            "stack_6_block0_output"],
                      'efficientnetlite3': [
                                            "stack_2_block2_output", 
                                            "stack_4_block4_output", 
                                            "stack_6_block0_output"],
                      'efficientnetlite3x': [
                                            "stack_2_block2_output", 
                                            "stack_4_block4_output", 
                                            "stack_6_block0_output"]
                      'efficientnetlite4': [
                                            "stack_2_block3_output", 
                                            "stack_4_block5_output", 
                                            "stack_6_block0_output"],
                      'resnet50': [
                                    "conv3_block4_out", 
                                    "conv4_block6_out", 
                                    "conv5_block3_out"],
                      'swin-tiny' : [
                                    "stack2_block2_output", 
                                    "stack3_block6_output",
                                    "stack4_block2_output",
                                     ]
                    }

        if config.BACKBONE in ['resnet50']:
            base_model = backbones[config.BACKBONE](
                            include_top=False,
                            weights='imagenet',
                            input_shape=config.IMAGE_SHAPE,
                        )
            outputs=[base_model.get_layer(x).output \
                    for x in out_layers[config.BACKBONE]]
        elif config.BACKBONE in ['swin-tiny']:
            base_model = backbones[config.BACKBONE](
                            pretrained='imagenet',
                            input_shape=config.IMAGE_SHAPE,
                        )
            outputs=[base_model.get_layer(x).output \
                     for x in out_layers[config.BACKBONE]]
        else:
            base_model = backbones[config.BACKBONE](
                                input_shape=config.IMAGE_SHAPE,
                                num_classes=0,
                                output_conv_filter=0,
                                activation='swish',
                            )
            outputs=[base_model.get_layer(x).output \
                    for x in out_layers[config.BACKBONE]]

            # Build additional input features that are not from backbone.
            for id in range(2): # Add p5->p6, p6->p7
                cur_name = "p{}_p{}_".format(id + 5, id + 6)
                additional_feature = tf.keras.layers.Conv2D(
                            config.W_BIFPN, 
                            kernel_size=1, 
                            name=cur_name + "channel_conv")(outputs[-1])
                additional_feature = tf.keras.layers.BatchNormalization(
                            epsilon=1e-3, 
                            name=cur_name + "channel_bn")(additional_feature)
                additional_feature = tf.keras.layers.MaxPool2D(
                            pool_size=3, 
                            strides=2, 
                            padding="SAME", 
                            name=cur_name + "max_down")(additional_feature)
                outputs.append(additional_feature)

        # whether to freeze the convolutional base
        base_model.trainable = config.BASE_MODEL_TRAINABLE 

        # Freeze BatchNormalization in pre-trained backbone
        if config.FREEZE_BACKBONE_BN:
          for layer in base_model.layers:
              if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        if not config.USE_FPN:
            for id in range(config.D_BIFPN):
                outputs = bi_fpn(
                                outputs, 
                                config.W_BIFPN, 
                                config.WEIGHTED_BIFPN, 
                                config.SEPARABLE_CONV, 
                                activation="swish", 
                                name="biFPN_{}_".format(id + 1))
                fpn_features = outputs
        else:
            fpn_features = build_FPN(outputs, config.FPN_FEATURE_MAP_SIZE)

        self.mask_head = MaskHead(config)

        anchorobj = anchor.Anchor(config=config)

        if not config.USE_FPN:
            # Outputs
            bboxes_features = det_header_pre(
                                            fpn_features, 
                                            config.W_BIFPN, 
                                            config.D_HEAD, 
                                            config.SEPARABLE_CONV, 
                                            activation="swish", 
                                            name="regressor_")
            box_net = det_header_post(
                                        bboxes_features,
                                        4, 
                                        config.ANCHOR_PER_PIX, 
                                        bias_init="zeros", 
                                        use_sep_conv=config.SEPARABLE_CONV, 
                                        head_activation=None, name="regressor_")

            if config.LOSS_CLASSIFICATION == "FOCAL":
                bias_init = initializers.constant(-math.log((1 - 0.01) / 0.01))
            else:
                bias_init = initializers.constant(0.0)
            class_features = det_header_pre(
                                            fpn_features, 
                                            config.W_BIFPN, 
                                            config.D_HEAD, 
                                            config.SEPARABLE_CONV, 
                                            activation="swish", 
                                            name="classifier_")
            if config.ACTIVATION == "SOFTMAX":
                num_classes = config.NUM_CLASSES+1
            else:
                num_classes = config.NUM_CLASSES
            class_net = det_header_post(
                                        class_features, 
                                        num_classes, 
                                        config.ANCHOR_PER_PIX, 
                                        bias_init, 
                                        config.SEPARABLE_CONV, 
                                        config.ACTIVATION, 
                                        name="classifier_")
            pred = {
                    'regression': box_net,
                    'classification': class_net,
                }
            # extract certain feature maps for FPN
            self.backbone = tf.keras.Model(inputs=base_model.input,
                                           outputs=pred)
        else:
            self.predictionHead = PredictionModule(config)
            # extract certain feature maps for FPN
            self.backbone = tf.keras.Model(inputs=base_model.input,
                                           outputs=fpn_features)


        self.num_anchors = anchorobj.num_anchors
        self.priors = anchorobj.anchors
        self.rescale = layers.Rescaling(scale=1. / 255)
        self.norm = layers.Normalization(
              mean=[0.485, 0.456, 0.406],
              variance=[0.229**2, 0.224**2, 0.225**2],
              axis=3,
          )

        # post-processing for evaluation
        self.detect = Detect(config=config)
        self.max_output_size = config.MAX_OUTPUT_SIZE
        self.num_classes = config.NUM_CLASSES
        self.config = config

    @tf.function
    def call(self, inputs, training=False):
        inputs, gt_boxes = inputs[0], inputs[1]

        if self.config.BACKBONE == 'resnet50':
            inputs = tf.keras.applications.resnet50.preprocess_input(inputs)
        elif self.config.BACKBONE in ['efficientnetlite0', 'efficientnetlite1', 
                                    'efficientnetlite2', 'efficientnetlite3', 
                                    'efficientnetlite3x', 'efficientnetlite4']:
            inputs = (inputs - 127.00) / 128.00
        else:
            inputs = self.rescale(inputs)
            inputs = self.norm(inputs)

        features = self.backbone(inputs, training=training)
        
        if not self.config.USE_FPN:
            classification = features['classification']
            regression = features['regression']
        else:
            # Prediction Head branch
            pred_cls = []
            pred_offset = []

            # all output from FPN use same prediction head
            for f_map in features:
                cls, offset = self.predictionHead(f_map)
                pred_cls.append(cls)
                pred_offset.append(offset)
                
            classification = tf.concat(pred_cls, axis=1)
            regression = tf.concat(pred_offset, axis=1)

        pred = {
            'regression': regression,
            'classification': classification,
            'priors': self.priors
        }

        pred.update(self.detect(pred, trad_nms=self.config.TRAD_NMS))

        if self.config.PREDICT_MASK:
            mask_feats = features[:-(self.config.TOTAL_FEAT_LAYERS - \
                                     self.config.MAX_MASK_FEAT_LAYER)]
            # Use features from C3 to C5 only, as shown in Ablation study in 
            # CenterMask
            if training:
                masks = self.mask_head(
                                       gt_boxes,
                                       mask_feats,
                                       self.num_classes,
                                       self.config,
                                       training)
            else:
                masks = self.mask_head(
                                       pred['detection_boxes'],
                                       mask_feats,
                                       self.num_classes,
                                       self.config,
                                       training)
            pred.update({'detection_masks': masks})

        return pred

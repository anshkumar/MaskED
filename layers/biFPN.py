import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from functools import reduce
import tensorflow_addons as tfa

BATCH_NORM_EPSILON = 1e-3

def activation_by_name(inputs, activation="relu", name=None):
    """Typical Activation layer added hard_swish and prelu."""
    if activation is None:
        return inputs

    layer_name = name and activation and name + activation
    activation_lower = activation.lower()
    if activation_lower == "hard_swish":
        return layers.Activation(activation=hard_swish, name=layer_name)(inputs)
    elif activation_lower == "mish":
        return layers.Activation(activation=mish, name=layer_name)(inputs)
    elif activation_lower == "phish":
        return layers.Activation(activation=phish, name=layer_name)(inputs)
    elif activation_lower == "prelu":
        shared_axes = list(range(1, len(inputs.shape)))
        shared_axes.pop(-1 if backend.image_data_format() == "channels_last" else 0)
        # print(f"{shared_axes = }")
        return layers.PReLU(shared_axes=shared_axes, alpha_initializer=initializers.Constant(0.25), name=layer_name)(inputs)
    elif activation_lower.startswith("gelu/app"):
        # gelu/approximate
        return tf.keras.activations.gelu(inputs, approximate=True, name=layer_name)
    elif activation_lower.startswith("gelu/linear"):
        return gelu_linear(inputs)
    elif activation_lower.startswith("leaky_relu/"):
        # leaky_relu with alpha parameter
        alpha = float(activation_lower.split("/")[-1])
        return layers.LeakyReLU(alpha=alpha, name=layer_name)(inputs)
    elif activation_lower == ("hard_sigmoid_torch"):
        return layers.Activation(activation=hard_sigmoid_torch, name=layer_name)(inputs)
    elif activation_lower == ("squaredrelu") or activation_lower == ("squared_relu"):
        return tf.math.pow(functional.tf.nn(inputs), 2)  # Squared ReLU: https://arxiv.org/abs/2109.08668
    elif activation_lower == ("starrelu") or activation_lower == ("star_relu"):
        from keras_cv_attention_models.nfnets.nfnets import ZeroInitGain

        # StarReLU: s * relu(x) ** 2 + b
        return ZeroInitGain(use_bias=True, weight_init_value=1.0, name=layer_name)(functional.pow(functional.relu(inputs), 2))
    else:
        return layers.Activation(activation=activation, name=layer_name)(inputs)

class ReluWeightedSum(layers.Layer):
    def __init__(self, initializer="ones", epsilon=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.initializer, self.epsilon = initializer, epsilon

    def build(self, input_shape):
        self.total = len(input_shape)
        self.gain = self.add_weight(name="gain", shape=(self.total,), initializer=self.initializer, dtype="float32", trainable=True)
        self.__epsilon__ = float(self.epsilon)
        super().build(input_shape)

    def call(self, inputs):
        gain = tf.nn.relu(self.gain)
        gain = gain / (tf.math.reduce_sum(gain) + self.__epsilon__)
        return tf.math.reduce_sum([inputs[id] * gain[id] for id in range(self.total)], axis=0)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"initializer": self.initializer, "epsilon": self.epsilon})
        return base_config

def align_feature_channel(inputs, output_channel, name=""):
    nn = inputs
    if inputs.shape[-1] != output_channel:
        nn = layers.Conv2D(output_channel, kernel_size=1, name=name + "channel_conv")(nn)
        nn = layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, name=name + "channel_bn")(nn)
    return nn

def resample_fuse(inputs, output_channel, use_weighted_sum=True, interpolation="nearest", use_sep_conv=True, activation="swish", name=""):
    inputs[0] = align_feature_channel(inputs[0], output_channel, name=name)

    if use_weighted_sum:
        nn = ReluWeightedSum(name=name + "wsm")(inputs)
    else:
        nn = layers.Activation(activation=activation, name=name + "sum")(inputs)
    nn = activation_by_name(nn, activation, name=name)
    if use_sep_conv:
        nn = layers.SeparableConv2D(output_channel, kernel_size=3, padding="SAME", use_bias=True, name=name + "sepconv")(nn)
    else:
        nn = layers.Conv2D(output_channel, kernel_size=3, padding="SAME", use_bias=True, name=name + "conv")(nn)
    nn = layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, name=name + "bn")(nn)
    return nn

def bi_fpn(features, output_channel, use_weighted_sum=True, use_sep_conv=True, interpolation="nearest", activation="swish", name=""):
    # print(f">>>> bi_fpn: {[ii.shape for ii in features] = }")
    # features: [p3, p4, p5, p6, p7]
    up_features = [features[-1]]
    for id, feature in enumerate(features[:-1][::-1]):
        cur_name = name + "p{}_up_".format(len(features) - id + 1)
        # up_feature = layers.UpSampling2D(size=(2, 2), interpolation=interpolation, name=cur_name + "up")(up_features[-1])
        size = tf.shape(feature)[1:-1]
        up_feature = tf.image.resize(up_features[-1], size, method=interpolation)
        up_feature = resample_fuse([feature, up_feature], output_channel, use_weighted_sum, use_sep_conv=use_sep_conv, activation=activation, name=cur_name)
        up_features.append(up_feature)
    # print(f">>>> bi_fpn: {[ii.shape for ii in up_features] = }")

    # up_features: [p7, p6_up, p5_up, p4_up, p3_up]
    out_features = [up_features[-1]]  # [p3_up]
    up_features = up_features[1:-1][::-1]  # [p4_up, p5_up, p6_up]
    for id, feature in enumerate(features[1:]):
        cur_name = name + "p{}_out_".format(len(features) - 1 + id)
        down_feature = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name=cur_name + "max_down")(out_features[-1])
        fusion_feature = [feature, down_feature] if id == len(up_features) else [feature, up_features[id], down_feature]
        out_feature = resample_fuse(fusion_feature, output_channel, use_weighted_sum, use_sep_conv=use_sep_conv, activation=activation, name=cur_name)
        out_features.append(out_feature)
    # out_features: [p3_up, p4_out, p5_out, p6_out, p7_out]
    return out_features

def det_header_pre(features, filters, depth, use_sep_conv=True, activation="swish", name=""):
    # print(f">>>> det_header_pre: {[ii.shape for ii in features] = }")
    if use_sep_conv:
        names = [name + "{}_sepconv".format(id + 1) for id in range(depth)]
        convs = [layers.SeparableConv2D(filters, kernel_size=3, padding="SAME", use_bias=True, name=names[id]) for id in range(depth)]
    else:
        names = [name + "{}_conv".format(id + 1) for id in range(depth)]
        convs = [layers.Conv2D(filters, kernel_size=3, padding="SAME", use_bias=True, name=names[id]) for id in range(depth)]

    outputs = []
    for feature_id, feature in enumerate(features):
        nn = feature
        for id in range(depth):
            nn = convs[id](nn)
            cur_name = name + "{}_{}_bn".format(id + 1, feature_id + 1)
            nn = layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, name=cur_name)(nn)
            nn = activation_by_name(nn, activation, name=cur_name + "{}_".format(id + 1))
        outputs.append(nn)
    return outputs


def det_header_post(inputs, classes=80, anchors=9, bias_init="zeros", use_sep_conv=True, head_activation="sigmoid", name=""):
    if use_sep_conv:
        header_conv = layers.SeparableConv2D(classes * anchors, kernel_size=3, padding="SAME", bias_initializer=bias_init, name=name + "head")
    else:
        header_conv = layers.Conv2D(classes * anchors, kernel_size=3, padding="SAME", bias_initializer=bias_init, name=name + "conv_head")
    outputs = [header_conv(ii) for ii in inputs]
    outputs = [layers.Reshape([-1, classes])(ii) for ii in outputs]
    outputs = tf.concat(outputs, axis=1)
    outputs = activation_by_name(outputs, head_activation, name=name + "output_")
    return outputs

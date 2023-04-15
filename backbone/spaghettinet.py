"""SpaghettiNet Feature Extractor."""
import collections
import tensorflow as tf
from tensorflow.python.training import moving_averages

IbnOp = collections.namedtuple(
    'IbnOp', ['kernel_size', 'expansion_rate', 'stride', 'has_residual'])
SepConvOp = collections.namedtuple('SepConvOp',
                                   ['kernel_size', 'stride', 'has_residual'])
IbnFusedGrouped = collections.namedtuple(
    'IbnFusedGrouped',
    ['kernel_size', 'expansion_rate', 'stride', 'groups', 'has_residual'])
SpaghettiStemNode = collections.namedtuple('SpaghettiStemNode',
                                           ['kernel_size', 'num_filters'])
SpaghettiNode = collections.namedtuple(
    'SpaghettiNode', ['layers', 'num_filters', 'edges', 'level'])
SpaghettiResampleEdge = collections.namedtuple('SpaghettiResampleEdge',
                                               ['input'])
SpaghettiPassthroughEdge = collections.namedtuple('SpaghettiPassthroughEdge',
                                                  ['input'])
SpaghettiNodeSpecs = collections.namedtuple('SpaghettiNodeSpecs',
                                            ['nodes', 'outputs'])

def conv_bn_act(out_channels, kernel_size, strides=1, groups=1, activation='selu', padding='same', name=None):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                groups=groups,
                use_bias=False,
                name="conv",
            ),
            tf.keras.layers.BatchNormalization(momentum=0.03, epsilon=0.001, name="bn"),
            tf.keras.layers.Activation(activation),
        ], name=name
    )

def dconv_bn_act(kernel_size, strides=1, activation='selu', padding='same', name=None):
    return tf.keras.Sequential(
        [
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=False,
                name="dconv",
            ),
            tf.keras.layers.BatchNormalization(momentum=0.03, epsilon=0.001, name="bn"),
            tf.keras.layers.Activation(activation),
        ], name=name
    )

def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.
  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.
  Args:
    tensor: A tensor of any type.
  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape

def nearest_neighbor_upsampling(input_tensor, scale=None, height_scale=None,
                                width_scale=None,
                                name='nearest_neighbor_upsampling'):
  """Nearest neighbor upsampling implementation.
  Nearest neighbor upsampling function that maps input tensor with shape
  [batch_size, height, width, channels] to [batch_size, height * scale
  , width * scale, channels]. This implementation only uses reshape and
  broadcasting to make it TPU compatible.
  Args:
    input_tensor: A float32 tensor of size [batch, height_in, width_in,
      channels].
    scale: An integer multiple to scale resolution of input data in both height
      and width dimensions.
    height_scale: An integer multiple to scale the height of input image. This
      option when provided overrides `scale` option.
    width_scale: An integer multiple to scale the width of input image. This
      option when provided overrides `scale` option.
    name: A name for the operation (optional).
  Returns:
    data_up: A float32 tensor of size
      [batch, height_in*scale, width_in*scale, channels].
  Raises:
    ValueError: If both scale and height_scale or if both scale and width_scale
      are None.
  """
  if not scale and (height_scale is None or width_scale is None):
    raise ValueError('Provide either `scale` or `height_scale` and'
                     ' `width_scale`.')
  with tf.name_scope(name):
    h_scale = scale if height_scale is None else height_scale
    w_scale = scale if width_scale is None else width_scale
    (batch_size, height, width,
     channels) = combined_static_and_dynamic_shape(input_tensor)
    output_tensor = tf.stack([input_tensor] * w_scale, axis=3, name='w_stack')
    # Adds a reshape op to avoid generating high-dimensional tensors that some
    # compilers cannot deal with.
    output_tensor = tf.reshape(output_tensor,
                               [batch_size, height, width * w_scale, channels])
    output_tensor = tf.stack([output_tensor] * h_scale, axis=2, name='h_stack')
    return tf.reshape(output_tensor,
                      [batch_size, height * h_scale, width * w_scale, channels])

def fixed_padding(inputs, kernel_size, rate=1):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    rate: An integer, rate for atrous convolution.
  Returns:
    output: A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
  pad_total = kernel_size_effective - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                  [pad_beg, pad_end], [0, 0]])
  return padded_inputs

class SpaghettiNet():
  """SpaghettiNet."""

  def __init__(self,
               node_specs,
               is_training=False,
               use_native_resize_op=False,
               use_explicit_padding=False,
               activation_fn=tf.nn.relu6,
               normalization_fn=tf.keras.layers.BatchNormalization,
               name='spaghetti_node'):
    self._node_specs = node_specs
    self._is_training = is_training
    self._use_native_resize_op = use_native_resize_op
    self._use_explicit_padding = use_explicit_padding
    self._activation_fn = activation_fn
    self._normalization_fn = normalization_fn
    self._name = name
    self._nodes = {}

  def _quant_var(self,
                 name,
                 initializer_val):
    """Create an var for storing the min/max quantization range."""
    return tf.Variable(
        name=name,
        shape=[],
        initial_value=initializer_val,
        trainable=False,
        dtype=tf.float32)

  def _quantizable_concat(self,
                          inputs,
                          axis,
                          is_training,
                          is_quantized=True,
                          default_min=0,
                          default_max=6,
                          ema_decay=0.999,
                          scope='quantized_concat'):
    """Concat replacement with quantization option.

    Allows concat inputs to share the same min max ranges,
    from experimental/gazelle/synthetic/model/tpu/utils.py.

    Args:
      inputs: list of tensors to concatenate.
      axis: dimension along which to concatenate.
      is_training: true if the graph is a training graph.
      is_quantized: flag to enable/disable quantization.
      default_min: default min value for fake quant op.
      default_max: default max value for fake quant op.
      ema_decay: the moving average decay for the quantization variables.
      scope: Optional scope for variable_scope.

    Returns:
      Tensor resulting from concatenation of input tensors
    """
    if is_quantized:
        min_var = self._quant_var('min', default_min)
        max_var = self._quant_var('max', default_max)
        if not is_training:
          # If we are building an eval graph just use the values in the
          # variables.
          quant_inputs = [
              tf.quantization.fake_quant_with_min_max_vars(t, min_var, max_var)
              for t in inputs
          ]
        else:
          concat_tensors = tf.concat(inputs, axis=axis)
          # TFLite requires that 0.0 is always in the [min; max] range.
          range_min = tf.minimum(
              tf.reduce_min(concat_tensors), 0.0, name='SafeQuantRangeMin')
          range_max = tf.maximum(
              tf.reduce_max(concat_tensors), 0.0, name='SafeQuantRangeMax')
          # Otherwise we need to keep track of the moving averages of the min
          # and of the elements of the input tensor max.
          min_val = moving_averages.assign_moving_average(
              min_var, range_min, ema_decay, name='AssignMinEma')
          max_val = moving_averages.assign_moving_average(
              max_var, range_max, ema_decay, name='AssignMaxEma')
          quant_inputs = [
              tf.quantization.fake_quant_with_min_max_vars(t, min_val, max_val)
              for t in inputs
          ]
        outputs = tf.concat(quant_inputs, axis=axis)
    else:
      outputs = tf.concat(inputs, axis=axis)
    return outputs

  def _expanded_conv(self, net, num_filters, expansion_rates, kernel_size,
                     stride, scope):
    """Expanded convolution."""
    expanded_num_filters = num_filters * expansion_rates
    add_fixed_padding = self._use_explicit_padding and stride > 1
    padding = 'VALID' if add_fixed_padding else 'SAME'
    net = conv_bn_act(
        expanded_num_filters, [1, 1],
        activation=self._activation_fn,
        padding=padding,
        name=scope + '/expansion')(net)
    net = dconv_bn_act(
        kernel_size=kernel_size,
        activation=self._activation_fn,
        strides=stride,
        padding=padding,
        name=scope + '/depthwise')(fixed_padding(net, kernel_size) if add_fixed_padding else net)
    net = conv_bn_act(
        num_filters, [1, 1],
        activation=tf.identity,
        padding=padding,
        name=scope + '/projection')(net)
    return net

  def _slice_shape_along_axis(self, shape, axis, groups):
    """Returns the shape after slicing into groups along the axis."""
    if isinstance(shape, tf.TensorShape):
      shape_as_list = shape.as_list()
      if shape_as_list[axis] % groups != 0:
        raise ValueError('Dimension {} must be divisible by {} groups'.format(
            shape_as_list[axis], groups))
      shape_as_list[axis] = shape_as_list[axis] // groups
      return tf.TensorShape(shape_as_list)
    elif isinstance(shape, tf.Tensor) and shape.shape.rank == 1:
      shape_as_list = tf.unstack(shape)
      shape_as_list[axis] = shape_as_list[axis] // groups
      return tf.stack(shape_as_list)
    else:
      raise ValueError(
          'Shape should be a TensorShape or rank-1 Tensor, but got: {}'.format(
              shape))

  def _ibn_fused_grouped(self, net, num_filters, expansion_rates, kernel_size,
                         stride, groups, scope):
    """Fused grouped IBN convolution."""
    add_fixed_padding = self._use_explicit_padding and stride > 1
    padding = 'VALID' if add_fixed_padding else 'SAME'
    slice_shape = self._slice_shape_along_axis(net.shape, -1, groups)
    slice_begin = [0] * net.shape.rank
    slice_outputs = []
    output_filters_per_group = net.shape[-1] // groups
    expanded_num_filters_per_group = output_filters_per_group * expansion_rates
    for idx in range(groups):
      slice_input = tf.slice(net, slice_begin, slice_shape)
      if isinstance(slice_shape, tf.TensorShape):
        slice_begin[-1] += slice_shape.as_list()[-1]
      else:
        slice_begin[-1] += slice_shape[-1]
      slice_outputs.append(
          conv_bn_act(
              expanded_num_filters_per_group,
              kernel_size,
              activation=self._activation_fn,
              strides=stride,
              padding=padding,
              name='{}/{}_{}'.format(scope, 'slice', idx))(fixed_padding(slice_input, kernel_size) if add_fixed_padding else slice_input) )
    # Make inputs to the concat share the same quantization variables.
    net = self._quantizable_concat(
        slice_outputs,
        -1,
        self._is_training,
        scope='{}/{}'.format(scope, 'concat'))
    net = conv_bn_act(
        num_filters, [1, 1],
        activation=tf.identity,
        padding=padding,
        name=scope + '/projection')(net)
    return net

  def _sep_conv(self, net, num_filters, kernel_size, stride, scope):
    """Depthwise Separable convolution."""
    add_fixed_padding = self._use_explicit_padding and stride > 1
    padding = 'VALID' if add_fixed_padding else 'SAME'
    net = dconv_bn_act(
        kernel_size=kernel_size,
        activation=None,
        strides=stride,
        padding=padding,
        name=scope + '/depthwise')(fixed_padding(net, kernel_size) if add_fixed_padding else net)
    net = conv_bn_act(
        num_filters, [1, 1],
        activation=self._activation_fn,
        padding=padding,
        name=scope + '/pointwise')(net)
    return net

  def _upsample(self, net, num_filters, upsample_ratio, scope):
    """Perform 1x1 conv then nearest neighbor upsampling."""
    node_pre_up = conv_bn_act(
        num_filters, [1, 1],
        activation=tf.identity,
        padding='SAME',
        name=scope + '/1x1_before_upsample')(net)
    if self._use_native_resize_op:
      with tf.name_scope(scope + '/nearest_neighbor_upsampling'):
        input_shape = combined_static_and_dynamic_shape(node_pre_up)
        node_up = tf.image.resize_nearest_neighbor(
            node_pre_up,
            [input_shape[1] * upsample_ratio, input_shape[2] * upsample_ratio])
    else:
      node_up = nearest_neighbor_upsampling(
          node_pre_up, scale=upsample_ratio)

    return node_up

  def _downsample(self, net, num_filters, downsample_ratio, scope):
    """Perform maxpool downsampling then 1x1 conv."""
    add_fixed_padding = self._use_explicit_padding and downsample_ratio > 1
    padding = 'VALID' if add_fixed_padding else 'SAME'
    node_down = tf.nn.max_pool2d(
        fixed_padding(net, downsample_ratio +
                          1) if add_fixed_padding else net,
        [downsample_ratio + 1, downsample_ratio + 1],
        strides=[downsample_ratio, downsample_ratio],
        padding=padding,
        name=scope + '/maxpool_downsampling')
    node_after_down = conv_bn_act(
        num_filters, [1, 1],
        activation=tf.identity,
        padding=padding,
        name=scope + '/1x1_after_downsampling')(node_down)
    return node_after_down

  def _no_resample(self, net, num_filters, scope):
    return conv_bn_act(
        num_filters, [1, 1],
        activation=tf.identity,
        padding='SAME',
        name=scope + '/1x1_no_resampling')(net)

  def _spaghetti_node(self, node, scope):
    """Spaghetti node."""
    node_spec = self._node_specs.nodes[node]

    # Make spaghetti edges
    edge_outputs = []
    edge_min_level = 100  # Currently we don't have any level over 7.
    edge_output_shape = None
    for edge in node_spec.edges:
      if isinstance(edge, SpaghettiPassthroughEdge):
        assert len(node_spec.edges) == 1, len(node_spec.edges)
        edge_outputs.append(self._nodes[edge.input])
      elif isinstance(edge, SpaghettiResampleEdge):
        edge_outputs.append(
            self._spaghetti_edge(node, edge.input,
                                 'edge_{}_{}'.format(edge.input, node)))
        if edge_min_level > self._node_specs.nodes[edge.input].level:
          edge_min_level = self._node_specs.nodes[edge.input].level
          edge_output_shape = tf.shape(edge_outputs[-1])
      else:
        raise ValueError('Unknown edge type {}'.format(edge))

    if len(edge_outputs) == 1:
      # When edge_outputs' length is 1, it is passthrough edge.
      net = edge_outputs[-1]
    else:
      # When edge_outputs' length is over 1, need to crop and then add edges.
      net = edge_outputs[0][:, :edge_output_shape[1], :edge_output_shape[2], :]
      for edge_output in edge_outputs[1:]:
        net += edge_output[:, :edge_output_shape[1], :edge_output_shape[2], :]
      net = self._activation_fn(net)

    # Make spaghetti node
    for idx, layer_spec in enumerate(node_spec.layers):
      if isinstance(layer_spec, IbnOp):
        net_exp = self._expanded_conv(net, node_spec.num_filters,
                                      layer_spec.expansion_rate,
                                      layer_spec.kernel_size, layer_spec.stride,
                                      '{}_{}'.format(scope, idx))
      elif isinstance(layer_spec, IbnFusedGrouped):
        net_exp = self._ibn_fused_grouped(net, node_spec.num_filters,
                                          layer_spec.expansion_rate,
                                          layer_spec.kernel_size,
                                          layer_spec.stride, layer_spec.groups,
                                          '{}_{}'.format(scope, idx))
      elif isinstance(layer_spec, SepConvOp):
        net_exp = self._sep_conv(net, node_spec.num_filters,
                                 layer_spec.kernel_size, layer_spec.stride,
                                 '{}_{}'.format(scope, idx))
      else:
        raise ValueError('Unsupported layer_spec: {}'.format(layer_spec))
      # Skip connection for all layers other than the first in a node.
      net = net_exp + net if layer_spec.has_residual else net_exp
    self._nodes[node] = net

  def _spaghetti_edge(self, curr_node, prev_node, scope):
    """Create an edge between curr_node and prev_node."""
    curr_spec = self._node_specs.nodes[curr_node]
    prev_spec = self._node_specs.nodes[prev_node]
    if curr_spec.level < prev_spec.level:
      # upsample
      output = self._upsample(self._nodes[prev_node], curr_spec.num_filters,
                              2**(prev_spec.level - curr_spec.level), scope)
    elif curr_spec.level > prev_spec.level:
      # downsample
      output = self._downsample(self._nodes[prev_node], curr_spec.num_filters,
                                2**(curr_spec.level - prev_spec.level), scope)
    else:
      # 1x1
      output = self._no_resample(self._nodes[prev_node], curr_spec.num_filters,
                                 scope)
    return output

  def _spaghetti_stem_node(self, net, node, scope):
    stem_spec = self._node_specs.nodes[node]
    kernel_size = stem_spec.kernel_size
    padding = 'VALID' if self._use_explicit_padding else 'SAME'
    self._nodes[node] = conv_bn_act(
        stem_spec.num_filters, [kernel_size, kernel_size],
        strides=2,
        activation=self._activation_fn,
        padding=padding,
        name=scope + '/stem')(fixed_padding(net, kernel_size) if self._use_explicit_padding else net)

  def apply(self, net, scope='spaghetti_net'):
    """Apply the SpaghettiNet to the input and return nodes in outputs."""
    for node, node_spec in self._node_specs.nodes.items():
      if isinstance(node_spec, SpaghettiStemNode):
        self._spaghetti_stem_node(net, node, '{}/stem_node'.format(scope))
      elif isinstance(node_spec, SpaghettiNode):
        self._spaghetti_node(node, '{}/{}'.format(scope, node))
      else:
        raise ValueError('Unknown node {}: {}'.format(node, node_spec))

    return [self._nodes[x] for x in self._node_specs.outputs]


def _spaghettinet_edgetpu_s():
  """Architecture definition for SpaghettiNet-EdgeTPU-S."""
  nodes = collections.OrderedDict()
  outputs = ['c0n1', 'c0n2', 'c0n3', 'c0n4', 'c0n5']
  nodes['s0'] = SpaghettiStemNode(kernel_size=5, num_filters=24)
  nodes['n0'] = SpaghettiNode(
      num_filters=48,
      level=2,
      layers=[
          IbnFusedGrouped(3, 8, 2, 3, False),
      ],
      edges=[SpaghettiPassthroughEdge(input='s0')])
  nodes['n1'] = SpaghettiNode(
      num_filters=64,
      level=3,
      layers=[
          IbnFusedGrouped(3, 4, 2, 4, False),
          IbnFusedGrouped(3, 4, 1, 4, True),
          IbnFusedGrouped(3, 4, 1, 4, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='n0')])
  nodes['n2'] = SpaghettiNode(
      num_filters=72,
      level=4,
      layers=[
          IbnOp(3, 8, 2, False),
          IbnFusedGrouped(3, 8, 1, 4, True),
          IbnOp(3, 8, 1, True),
          IbnOp(3, 4, 1, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='n1')])
  nodes['n3'] = SpaghettiNode(
      num_filters=88,
      level=5,
      layers=[
          IbnOp(3, 8, 2, False),
          IbnOp(3, 8, 1, True),
          IbnOp(3, 4, 1, True),
          IbnOp(3, 4, 1, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='n2')])
  nodes['n4'] = SpaghettiNode(
      num_filters=88,
      level=6,
      layers=[
          IbnOp(3, 8, 2, False),
          SepConvOp(5, 1, True),
          SepConvOp(5, 1, True),
          SepConvOp(5, 1, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='n3')])
  nodes['n5'] = SpaghettiNode(
      num_filters=88,
      level=7,
      layers=[
          SepConvOp(5, 2, False),
          SepConvOp(3, 1, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='n4')])
  nodes['c0n0'] = SpaghettiNode(
      num_filters=144,
      level=5,
      layers=[
          IbnOp(3, 4, 1, False),
          IbnOp(3, 4, 1, True),
          IbnOp(3, 4, 1, True),
          IbnOp(3, 4, 1, True),
      ],
      edges=[
          SpaghettiResampleEdge(input='n3'),
          SpaghettiResampleEdge(input='n4')
      ])
  nodes['c0n1'] = SpaghettiNode(
      num_filters=120,
      level=4,
      layers=[
          IbnOp(3, 8, 1, False),
          IbnOp(3, 4, 1, True),
          IbnOp(3, 4, 1, True),
          IbnOp(3, 4, 1, True),
      ],
      edges=[
          SpaghettiResampleEdge(input='n2'),
          SpaghettiResampleEdge(input='c0n0')
      ])
  nodes['c0n2'] = SpaghettiNode(
      num_filters=168,
      level=5,
      layers=[
          IbnOp(3, 4, 1, False),
      ],
      edges=[
          SpaghettiResampleEdge(input='c0n1'),
          SpaghettiResampleEdge(input='c0n0')
      ])
  nodes['c0n3'] = SpaghettiNode(
      num_filters=136,
      level=6,
      layers=[
          IbnOp(3, 4, 1, False),
          SepConvOp(3, 1, True),
      ],
      edges=[
          SpaghettiResampleEdge(input='n5'),
          SpaghettiResampleEdge(input='c0n0')
      ])
  nodes['c0n4'] = SpaghettiNode(
      num_filters=136,
      level=7,
      layers=[
          IbnOp(3, 4, 1, False),
      ],
      edges=[
          SpaghettiResampleEdge(input='n5'),
          SpaghettiResampleEdge(input='c0n0')
      ])
  nodes['c0n5'] = SpaghettiNode(
      num_filters=64,
      level=8,
      layers=[
          SepConvOp(3, 1, False),
          SepConvOp(3, 1, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='c0n4')])
  node_specs = SpaghettiNodeSpecs(nodes=nodes, outputs=outputs)
  return node_specs


def _spaghettinet_edgetpu_m():
  """Architecture definition for SpaghettiNet-EdgeTPU-M."""
  nodes = collections.OrderedDict()
  outputs = ['c0n1', 'c0n2', 'c0n3', 'c0n4', 'c0n5']
  nodes['s0'] = SpaghettiStemNode(kernel_size=5, num_filters=24)
  nodes['n0'] = SpaghettiNode(
      num_filters=48,
      level=2,
      layers=[
          IbnFusedGrouped(3, 8, 2, 3, False),
      ],
      edges=[SpaghettiPassthroughEdge(input='s0')])
  nodes['n1'] = SpaghettiNode(
      num_filters=64,
      level=3,
      layers=[
          IbnFusedGrouped(3, 8, 2, 4, False),
          IbnFusedGrouped(3, 4, 1, 4, True),
          IbnFusedGrouped(3, 4, 1, 4, True),
          IbnFusedGrouped(3, 4, 1, 4, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='n0')])
  nodes['n2'] = SpaghettiNode(
      num_filters=72,
      level=4,
      layers=[
          IbnOp(3, 8, 2, False),
          IbnFusedGrouped(3, 8, 1, 4, True),
          IbnOp(3, 8, 1, True),
          IbnOp(3, 8, 1, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='n1')])
  nodes['n3'] = SpaghettiNode(
      num_filters=96,
      level=5,
      layers=[
          IbnOp(3, 8, 2, False),
          IbnOp(3, 8, 1, True),
          IbnOp(3, 8, 1, True),
          IbnOp(3, 4, 1, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='n2')])
  nodes['n4'] = SpaghettiNode(
      num_filters=104,
      level=6,
      layers=[
          IbnOp(3, 8, 2, False),
          IbnOp(3, 4, 1, True),
          SepConvOp(5, 1, True),
          IbnOp(3, 4, 1, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='n3')])
  nodes['n5'] = SpaghettiNode(
      num_filters=56,
      level=7,
      layers=[
          SepConvOp(5, 2, False),
          SepConvOp(3, 1, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='n4')])
  nodes['c0n0'] = SpaghettiNode(
      num_filters=152,
      level=5,
      layers=[
          IbnOp(3, 8, 1, False),
          IbnOp(3, 4, 1, True),
          IbnOp(3, 4, 1, True),
          IbnOp(3, 4, 1, True),
      ],
      edges=[
          SpaghettiResampleEdge(input='n3'),
          SpaghettiResampleEdge(input='n4')
      ])
  nodes['c0n1'] = SpaghettiNode(
      num_filters=120,
      level=4,
      layers=[
          IbnOp(3, 8, 1, False),
          IbnOp(3, 4, 1, True),
          IbnOp(3, 4, 1, True),
          IbnOp(3, 4, 1, True),
      ],
      edges=[
          SpaghettiResampleEdge(input='n2'),
          SpaghettiResampleEdge(input='c0n0')
      ])
  nodes['c0n2'] = SpaghettiNode(
      num_filters=168,
      level=5,
      layers=[
          IbnOp(3, 4, 1, False),
          SepConvOp(3, 1, True),
      ],
      edges=[
          SpaghettiResampleEdge(input='c0n1'),
          SpaghettiResampleEdge(input='c0n0')
      ])
  nodes['c0n3'] = SpaghettiNode(
      num_filters=136,
      level=6,
      layers=[
          SepConvOp(3, 1, False),
          SepConvOp(3, 1, True),
          SepConvOp(3, 1, True),
      ],
      edges=[
          SpaghettiResampleEdge(input='n5'),
          SpaghettiResampleEdge(input='c0n0')
      ])
  nodes['c0n4'] = SpaghettiNode(
      num_filters=136,
      level=7,
      layers=[
          IbnOp(3, 4, 1, False),
          SepConvOp(5, 1, True),
      ],
      edges=[
          SpaghettiResampleEdge(input='n5'),
          SpaghettiResampleEdge(input='c0n0')
      ])
  nodes['c0n5'] = SpaghettiNode(
      num_filters=64,
      level=8,
      layers=[
          SepConvOp(3, 1, False),
          SepConvOp(3, 1, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='c0n4')])
  node_specs = SpaghettiNodeSpecs(nodes=nodes, outputs=outputs)
  return node_specs


def _spaghettinet_edgetpu_l():
  """Architecture definition for SpaghettiNet-EdgeTPU-L."""
  nodes = collections.OrderedDict()
  outputs = ['c0n1', 'c0n2', 'c0n3', 'c0n4', 'c0n5']
  nodes['s0'] = SpaghettiStemNode(kernel_size=5, num_filters=24)
  nodes['n0'] = SpaghettiNode(
      num_filters=48,
      level=2,
      layers=[
          IbnFusedGrouped(3, 8, 2, 3, False),
      ],
      edges=[SpaghettiPassthroughEdge(input='s0')])
  nodes['n1'] = SpaghettiNode(
      num_filters=64,
      level=3,
      layers=[
          IbnFusedGrouped(3, 8, 2, 4, False),
          IbnFusedGrouped(3, 8, 1, 4, True),
          IbnFusedGrouped(3, 8, 1, 4, True),
          IbnFusedGrouped(3, 4, 1, 4, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='n0')])
  nodes['n2'] = SpaghettiNode(
      num_filters=80,
      level=4,
      layers=[
          IbnOp(3, 8, 2, False),
          IbnOp(3, 8, 1, True),
          IbnOp(3, 8, 1, True),
          IbnOp(3, 4, 1, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='n1')])
  nodes['n3'] = SpaghettiNode(
      num_filters=104,
      level=5,
      layers=[
          IbnOp(3, 8, 2, False),
          IbnOp(3, 8, 1, True),
          IbnOp(3, 8, 1, True),
          IbnOp(3, 8, 1, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='n2')])
  nodes['n4'] = SpaghettiNode(
      num_filters=88,
      level=6,
      layers=[
          IbnOp(3, 8, 2, False),
          IbnOp(5, 4, 1, True),
          IbnOp(3, 4, 1, True),
          IbnOp(3, 8, 1, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='n3')])
  nodes['n5'] = SpaghettiNode(
      num_filters=56,
      level=7,
      layers=[
          IbnOp(5, 4, 2, False),
          SepConvOp(5, 1, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='n4')])
  nodes['c0n0'] = SpaghettiNode(
      num_filters=160,
      level=5,
      layers=[
          IbnOp(3, 8, 1, False),
          IbnOp(3, 4, 1, True),
          IbnOp(3, 4, 1, True),
          IbnOp(3, 4, 1, True),
      ],
      edges=[
          SpaghettiResampleEdge(input='n3'),
          SpaghettiResampleEdge(input='n4')
      ])
  nodes['c0n1'] = SpaghettiNode(
      num_filters=120,
      level=4,
      layers=[
          IbnOp(3, 8, 1, False),
          IbnOp(3, 4, 1, True),
          IbnOp(3, 4, 1, True),
          IbnOp(3, 8, 1, True),
      ],
      edges=[
          SpaghettiResampleEdge(input='n2'),
          SpaghettiResampleEdge(input='c0n0')
      ])
  nodes['c0n2'] = SpaghettiNode(
      num_filters=168,
      level=5,
      layers=[
          IbnOp(3, 4, 1, False),
          IbnOp(3, 4, 1, True),
          IbnOp(3, 4, 1, True),
          IbnOp(3, 4, 1, True),
      ],
      edges=[
          SpaghettiResampleEdge(input='c0n1'),
          SpaghettiResampleEdge(input='c0n0')
      ])
  nodes['c0n3'] = SpaghettiNode(
      num_filters=112,
      level=6,
      layers=[
          IbnOp(3, 8, 1, False),
          IbnOp(3, 4, 1, True),
          SepConvOp(3, 1, True),
      ],
      edges=[
          SpaghettiResampleEdge(input='n5'),
          SpaghettiResampleEdge(input='c0n0')
      ])
  nodes['c0n4'] = SpaghettiNode(
      num_filters=128,
      level=7,
      layers=[
          IbnOp(3, 4, 1, False),
          IbnOp(3, 4, 1, True),
      ],
      edges=[
          SpaghettiResampleEdge(input='n5'),
          SpaghettiResampleEdge(input='c0n0')
      ])
  nodes['c0n5'] = SpaghettiNode(
      num_filters=64,
      level=8,
      layers=[
          SepConvOp(5, 1, False),
          SepConvOp(5, 1, True),
      ],
      edges=[SpaghettiPassthroughEdge(input='c0n4')])
  node_specs = SpaghettiNodeSpecs(nodes=nodes, outputs=outputs)
  return node_specs


def lookup_spaghetti_arch(arch):
  """Lookup table for the nodes structure for spaghetti nets."""
  if arch == 'spaghettinet_edgetpu_s':
    return _spaghettinet_edgetpu_s()
  elif arch == 'spaghettinet_edgetpu_m':
    return _spaghettinet_edgetpu_m()
  elif arch == 'spaghettinet_edgetpu_l':
    return _spaghettinet_edgetpu_l()
  else:
    raise ValueError('Unknown architecture {}'.format(arch))


# class SSDSpaghettinetFeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
#   """SSD Feature Extractor using Custom Architecture."""

#   def __init__(
#       self,
#       is_training,
#       depth_multiplier,
#       min_depth,
#       pad_to_multiple,
#       conv_hyperparams_fn,
#       spaghettinet_arch_name='spaghettinet_edgetpu_m',
#       use_explicit_padding=False,
#       reuse_weights=False,
#       use_depthwise=False,
#       override_base_feature_extractor_hyperparams=False,
#   ):
#     """SSD FPN feature extractor based on Mobilenet v2 architecture.

#     Args:
#       is_training: whether the network is in training mode.
#       depth_multiplier: Not used in SpaghettiNet.
#       min_depth: Not used in SpaghettiNet.
#       pad_to_multiple: Not used in SpaghettiNet.
#       conv_hyperparams_fn: Not used in SpaghettiNet.
#       spaghettinet_arch_name: name of the specific architecture.
#       use_explicit_padding: Whether to use explicit padding when extracting
#         features. Default is False.
#       reuse_weights: Not used in SpaghettiNet.
#       use_depthwise: Not used in SpaghettiNet.
#       override_base_feature_extractor_hyperparams: Not used in SpaghettiNet.
#     """
#     super(SSDSpaghettinetFeatureExtractor, self).__init__(
#         is_training=is_training,
#         use_explicit_padding=use_explicit_padding,
#         depth_multiplier=depth_multiplier,
#         min_depth=min_depth,
#         pad_to_multiple=pad_to_multiple,
#         conv_hyperparams_fn=conv_hyperparams_fn,
#         reuse_weights=reuse_weights,
#         use_depthwise=use_depthwise,
#         override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams
#     )
#     self._spaghettinet_arch_name = spaghettinet_arch_name
#     self._use_native_resize_op = False if is_training else True

#   def preprocess(self, resized_inputs):
#     """SSD preprocessing.

#     Maps pixel values to the range [-1, 1].

#     Args:
#       resized_inputs: a [batch, height, width, channels] float tensor
#         representing a batch of images.

#     Returns:
#       preprocessed_inputs: a [batch, height, width, channels] float tensor
#         representing a batch of images.
#     """
#     return (2.0 / 255.0) * resized_inputs - 1.0

#   def extract_features(self, preprocessed_inputs):
#     """Extract features from preprocessed inputs.

#     Args:
#       preprocessed_inputs: a [batch, height, width, channels] float tensor
#         representing a batch of images.

#     Returns:
#       feature_maps: a list of tensors where the ith tensor has shape
#         [batch, height_i, width_i, depth_i]
#     """
#     nodes_dict = lookup_spaghetti_arch(self._spaghettinet_arch_name)

#     with tf.variable_scope(
#         self._spaghettinet_arch_name, reuse=self._reuse_weights):
#       with slim.arg_scope([tf.nn.conv2d],
#                           weights_initializer=tf.truncated_normal_initializer(
#                               mean=0.0, stddev=0.03),
#                           weights_regularizer=slim.l2_regularizer(1e-5)):
#         with slim.arg_scope([tf.nn.depthwise_conv2d],
#                             weights_initializer=tf.truncated_normal_initializer(
#                                 mean=0.0, stddev=0.03),
#                             weights_regularizer=slim.l2_regularizer(1e-5)):
#           with slim.arg_scope([slim.batch_norm],
#                               is_training=self._is_training,
#                               epsilon=0.001,
#                               decay=0.97,
#                               center=True,
#                               scale=True):
#             spaghetti_net = SpaghettiNet(
#                 node_specs=nodes_dict,
#                 is_training=self._is_training,
#                 use_native_resize_op=self._use_native_resize_op,
#                 use_explicit_padding=self._use_explicit_padding,
#                 name=self._spaghettinet_arch_name)
#             feature_maps = spaghetti_net.apply(preprocessed_inputs)
#     return feature_maps

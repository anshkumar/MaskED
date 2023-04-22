from itertools import product
from math import sqrt
from utils import utils
import tensorflow as tf
import numpy as np

# Can generate one instance only when creating the model
class Anchor(object):

    def __init__(self, config):
        """
        :param img_size:
        :param feature_map_size:
        :param aspect_ratio:
        :param scale:
        """
        self.config = config
        self.anchors_norm = self._generate_anchors(
            config.IMAGE_SHAPE, 
            [config.MIN_MASK_FEAT_LAYER, config.TOTAL_FEAT_LAYERS], 
            config.ANCHOR_RATIOS,
            config.NUM_SCALES,
            config.ANCHOR_SCALE
            )
        self.anchors = self.get_anchors()

    def get_feature_sizes(self, input_shape, pyramid_levels=[3, 7]):
        #https://github.com/google/automl/tree/master/efficientdet/utils.py#L509
        feature_sizes = [input_shape[:2]]
        for _ in range(max(pyramid_levels)):
            pre_feat_size = feature_sizes[-1]
            feature_sizes.append(((pre_feat_size[0] - 1) // 2 + 1, 
                                    (pre_feat_size[1] - 1) // 2 + 1))
        return feature_sizes

    def _generate_anchors(self, input_shape=(512, 512, 3), 
        pyramid_levels=[3, 7], aspect_ratios=[1, 2, 0.5], num_scales=3, 
        anchor_scale=4, grid_zero_start=False):
        # base anchors
        scales = [2 ** (ii / num_scales) * anchor_scale \
                for ii in range(num_scales)]
        aspect_ratios_tensor = np.array(aspect_ratios, dtype="float32")
        if len(aspect_ratios_tensor.shape) == 1:
            # aspect_ratios = [0.5, 1, 2]
            sqrt_ratios = np.sqrt(aspect_ratios_tensor)
            ww_ratios, hh_ratios = sqrt_ratios, 1 / sqrt_ratios
        else:
            # aspect_ratios = [(1, 1), (1.4, 0.7), (0.7, 1.4)]
            ww_ratios, hh_ratios = aspect_ratios_tensor[:, 0], \
                                   aspect_ratios_tensor[:, 1]
        base_anchors_hh = np.reshape(np.expand_dims(scales, 1) * \
                          np.expand_dims(hh_ratios, 0), [-1])
        base_anchors_ww = np.reshape(np.expand_dims(scales, 1) * \
                          np.expand_dims(ww_ratios, 0), [-1])
        base_anchors_hh_half, base_anchors_ww_half = base_anchors_hh / 2, \
                                                     base_anchors_ww / 2
        base_anchors = np.stack([
            base_anchors_hh_half * -1, base_anchors_ww_half * -1, 
            base_anchors_hh_half, base_anchors_ww_half], axis=1)
        # re-order according to official generated anchors
        # base_anchors = tf.gather(base_anchors, [3, 6, 0, 4, 7, 1, 5, 8, 2])  

        # make grid
        pyramid_levels = list(range(min(pyramid_levels), 
                                    max(pyramid_levels) + 1))
        feature_sizes = self.get_feature_sizes(input_shape, pyramid_levels)

        all_anchors = []
        for level in pyramid_levels:
            stride_hh, stride_ww = feature_sizes[0][0]/feature_sizes[level][0],\
                                   feature_sizes[0][1]/feature_sizes[level][1]
            top, left = (0, 0) if grid_zero_start else (stride_hh / 2, 
                                                        stride_ww / 2)
            hh_centers = np.arange(top, input_shape[0], stride_hh)
            ww_centers = np.arange(left, input_shape[1], stride_ww)
            ww_grid, hh_grid = np.meshgrid(ww_centers, hh_centers)
            grid = np.reshape(np.stack([hh_grid, ww_grid, hh_grid, ww_grid], 2), 
                              [-1, 1, 4])
            anchors = np.expand_dims(base_anchors * \
                            [stride_hh, stride_ww, stride_hh, stride_ww], 0) + \
                      grid.astype(base_anchors.dtype)
            anchors = np.reshape(anchors, [-1, 4])
            all_anchors.append(anchors)
        all_anchors = np.concatenate(all_anchors, axis=0) / \
                [input_shape[0], input_shape[1], input_shape[0], input_shape[1]]

        return all_anchors

    def get_anchors(self):
        # Convert anchors from [cx, cy, w, h] to [ymin, xmin, ymax, xmax ] 
        # for IOU calculations
        w = self.anchors_norm[:, 2]
        h = self.anchors_norm[:, 3]
        anchors_yxyx = tf.cast(tf.stack(
            [(self.anchors_norm[:, 1] - (h / 2)), 
            (self.anchors_norm[:, 0] - (w / 2)), 
            (self.anchors_norm[:, 1] + (h / 2)), 
            (self.anchors_norm[:, 0] + (w / 2))], 
            axis=-1), tf.float32)

        return anchors_yxyx

    def matching(self, pos_thresh, neg_thresh, gt_bbox, gt_labels, config):
        # size: [num_objects, num_priors]; anchors along the row and 
        # ground_truth clong the columns

        # anchors and gt_bbox in [y1, x1, y2, x2]
        pairwise_iou = utils._iou(self.anchors, gt_bbox) 

        # size [num_priors]; iou with ground truth with the anchors
        each_prior_max = tf.reduce_max(pairwise_iou, axis=-1) 

        if tf.shape(pairwise_iou)[-1] == 0: # No positive ground-truth boxes
          return (self.anchors*0, tf.cast(self.anchors[:, 0]*0, dtype=tf.int64), 
                self.anchors*0, tf.cast(self.anchors[:, 0]*0, dtype=tf.int64))

        # size [num_priors]; id of groud truth having max iou with the anchors
        each_prior_index = tf.math.argmax(pairwise_iou, axis=-1) 

        each_box_max = tf.reduce_max(pairwise_iou, axis=0)
        each_box_index = tf.math.argmax(pairwise_iou, axis=0)

        # For the max IoU prior for each gt box, set its IoU to 2. This ensures 
        # that it won't be filtered in the threshold step even if the IoU is 
        # under the negative threshold. This is because that we want
        # at least one prior to match with each gt box or else we'd be wasting 
        # training data.

        indices = tf.expand_dims(each_box_index,axis=-1)

        updates = tf.cast(tf.tile(tf.constant([2]), tf.shape(each_box_index)), 
            dtype=tf.float32)
        each_prior_max = tf.tensor_scatter_nd_update(each_prior_max, indices, 
            updates)

        # Set the index of the pair (prior, gt) we set the overlap for above.
        updates = tf.cast(tf.range(0,tf.shape(each_box_index)[0]),
            dtype=tf.int64)
        each_prior_index = tf.tensor_scatter_nd_update(each_prior_index, 
            indices, updates)

        # size: [num_priors, 4]; each_prior_box in [y1, x1, y2, x2]
        each_prior_box = tf.gather(gt_bbox, each_prior_index) 

        # the class of the max IoU gt box for each prior, size: [num_priors]
        conf = tf.squeeze(tf.gather(gt_labels, each_prior_index)) 

        neutral_label_index = tf.where(each_prior_max < pos_thresh)
        background_label_index = tf.where(each_prior_max < neg_thresh)

        conf = tf.tensor_scatter_nd_update(conf, 
            neutral_label_index, 
            -1*tf.ones(tf.size(neutral_label_index), dtype=tf.int64))
        conf = tf.tensor_scatter_nd_update(conf, 
            background_label_index, 
            tf.zeros(tf.size(background_label_index), dtype=tf.int64))

        # anchors and each_prior_box in [y1, x1, y2, x2]
        offsets = utils._encode(
            each_prior_box, 
            self.anchors, 
            include_variances=config.INCLUDE_VARIANCES)

        return offsets, conf, each_prior_box, each_prior_index

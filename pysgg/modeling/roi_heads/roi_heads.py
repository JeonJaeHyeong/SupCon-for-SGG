# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .attribute_head.attribute_head import build_roi_attribute_head
from .box_head.box_head import build_roi_box_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .mask_head.mask_head import build_roi_mask_head
from .relation_head.relation_head import build_roi_relation_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None, logger=None):
        losses = {}

        # proposals : [BoxList(num_boxes=10...mode=xyxy), BoxList(num_boxes=10...mode=xyxy), ... ], len(6)
        # features[0].shape : torch.Size([6, 256, 152, 256])
        # features[1].shape : torch.Size([6, 256, 76, 128])
        # features[2].shape : torch.Size([6, 256, 38, 64])
        # features[3].shape : torch.Size([6, 256, 19, 32])
        # features[4].shape : ttorch.Size([6, 256, 10, 16]
        # targets : [BoxList(num_boxes=14...mode=xyxy), BoxList(num_boxes=9,...mode=xyxy), ...], len(6)

        # x.shape : torch.Size([480, 4096])
        # detections : [BoxList(num_boxes=80...mode=xyxy), BoxList(num_boxes=80...mode=xyxy), ... ], len(6)
        # loss_box : {}
        x, detections, loss_box = self.box(features, proposals, targets)
        if not self.cfg.MODEL.RELATION_ON: # not true : false
            # During the relationship training stage, the bbox_proposal_network should be fixed, and no loss. 
            losses.update(loss_box)

        if self.cfg.MODEL.ATTRIBUTE_ON: # false
            # Attribute head don't have a separate feature extractor
            z, detections, loss_attribute = self.attribute(features, detections, targets)
            losses.update(loss_attribute)

        if self.cfg.MODEL.MASK_ON: # false
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON: # false
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)

        if self.cfg.MODEL.RELATION_ON: # true
            # it may be not safe to share features due to post processing
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            
            # len(featuers) : 5,    features[0].shape : torch.Size([6, 256, 152, 256])
            # len(detections) : 6,  detections : [BoxList(num_boxes=80...mode=xyxy), BoxList(num_boxes=80...mode=xyxy), ... ]
            # len(targets) : 6,     targets : [BoxList(num_boxes=14...mode=xyxy), BoxList(num_boxes=9,...mode=xyxy), ... ]
            # self.relation : ROIRelationHead
            x, detections, loss_relation = self.relation(features, detections, targets, logger)
            # loss_relation
            # {'loss_rel': tensor(0.2078, devic...Backward>), 
            # 'pre_rel_classify_loss_iter-0': tensor(1.1105, devic...ackward0>), 
            # 'pre_rel_classify_loss_iter-1': tensor(0.5408, devic...ackward0>), 
            # 'pre_rel_classify_loss_iter-2': tensor(0.6627, devic...ackward0>)}
            # losses : {}
            losses.update(loss_relation)

        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON: # false
        return []

    if not cfg.MODEL.RPN_ONLY: # not false = true
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON: # false
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON: # false
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))
    if cfg.MODEL.RELATION_ON: # true
        roi_heads.append(("relation", build_roi_relation_head(cfg, in_channels)))
    if cfg.MODEL.ATTRIBUTE_ON: # false
        roi_heads.append(("attribute", build_roi_attribute_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads

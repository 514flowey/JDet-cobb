from .convfc_roi_head import ConvFCRoIHead

from jdet.utils.registry import build_from_cfg, LOSSES, HEADS, BOXES
from jdet.ops.bbox_transforms import hbb2obb
from jdet.ops.nms_rotated import multiclass_nms_rotated
from jdet.models.boxes.box_ops import rotated_box_to_poly, rotated_box_to_bbox, poly_to_rotated_box
from jdet.utils.general import multi_apply

from jittor import nn
import jittor as jt

@HEADS.register_module()
class COBBRoIHead(ConvFCRoIHead):
    def __init__(self,
                 num_score_convs=0,
                 num_score_fcs=0,
                 num_ratio_convs=0,
                 num_ratio_fcs=0,
                 loss_score=dict(
                     type='SmoothL1Loss', 
                     beta=1.0 / 3.0, 
                     loss_weight=1.0,
                     ),
                 loss_ratio=dict(
                     type='SmoothL1Loss', 
                     beta=1.0 / 3.0, 
                     loss_weight=16.0
                     ),
                 cobb_coder=None,
                 score_type='sigmoid',
                 ratio_type='sigmoid',
                 score_dim=4,
                 ratio_dim=1,
                 **kwargs):

        self.score_type = score_type
        self.ratio_type = ratio_type
        self.score_dim = score_dim
        self.ratio_dim = ratio_dim

        self.num_score_convs = num_score_convs
        self.num_score_fcs = num_score_fcs
        self.num_ratio_convs = num_ratio_convs
        self.num_ratio_fcs = num_ratio_fcs

        self.loss_score = build_from_cfg(loss_score, LOSSES)
        self.loss_ratio = build_from_cfg(loss_ratio, LOSSES)
        self.cobb_coder = build_from_cfg(cobb_coder, BOXES)
        super(COBBRoIHead, self).__init__(**kwargs)

    def _init_layers(self):
        super(COBBRoIHead, self)._init_layers()
        self.score_convs, self.score_fcs, self.score_last_dim = \
            self._add_conv_fc_branch(
                self.num_score_convs, self.num_score_fcs, self.shared_out_channels)
        self.ratio_convs, self.ratio_fcs, self.ratio_last_dim = \
            self._add_conv_fc_branch(
                self.num_ratio_convs, self.num_ratio_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_score_fcs == 0:
                self.score_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_ratio_fcs == 0:
                self.ratio_last_dim *= (self.roi_feat_size * self.roi_feat_size)

        out_dim_score = self.score_dim if self.reg_class_agnostic else self.score_dim * self.num_classes
        self.fc_score = nn.Linear(self.score_last_dim, out_dim_score)

        out_dim_ratio = self.ratio_dim if self.reg_class_agnostic else self.num_classes
        self.fc_ratio = nn.Linear(self.score_last_dim, out_dim_ratio)

    def init_weights(self):
        super(COBBRoIHead, self).init_weights()
        for module_list in [self.score_fcs, self.ratio_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

        nn.init.gauss_(self.fc_score.weight, 0, 0.001)
        nn.init.constant_(self.fc_score.bias, 0)

        nn.init.gauss_(self.fc_ratio.weight, 0, 0.001)
        nn.init.constant_(self.fc_ratio.bias, 0)

    def get_targets_gt_parse(self, target):
        hboxes = target['hboxes']
        rboxes = target['rboxes']
        return hboxes, rboxes

    def forward(self, x):
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        x_cls = x
        x_reg = x
        x_score = x
        x_ratio = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.ndim > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.ndim > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        
        for conv in self.score_convs:
            x_score = conv(x_score)
        if x_score.ndim > 2:
            if self.with_avg_pool:
                x_score = self.avg_pool(x_score)
            x_score = x_score.view(x_score.size(0), -1)
        for fc in self.score_fcs:
            x_score = self.relu(fc(x_score))

        for conv in self.ratio_convs:
            x_ratio = conv(x_ratio)
        if x_ratio.ndim > 2:
            if self.with_avg_pool:
                x_ratio = self.avg_pool(x_ratio)
            x_ratio = x_ratio.view(x_ratio.size(0), -1)
        for fc in self.ratio_fcs:
            x_ratio = self.relu(fc(x_ratio))

        cls_score = self.fc_cls(x_cls)
        bbox_pred = self.fc_reg(x_reg)
        if self.score_type == 'cos':
            score_pred = jt.cos(self.fc_score(x)) * 0.5 + 0.5
        elif self.score_type == 'sigmoid':
            score_pred = jt.sigmoid(self.fc_score(x))
        elif self.score_type == 'add0.5':
            score_pred = self.fc_score(x) + 0.5
        elif self.score_type == 'softmax':
            score_pred = jt.softmax(self.fc_score(x), dim=-1)
        else:
            score_pred = self.fc_score(x)
        if self.ratio_type == 'cos':
            ratio_pred = jt.cos(self.fc_ratio(x)) * 0.5 + 0.5
        elif self.ratio_type ==' sigmoid':
            ratio_pred = jt.sigmoid(self.fc_ratio(x))
        elif self.ratio_type == 'add0.5':
            ratio_pred = self.fc_ratio(x) + 0.5
        else:
            ratio_pred = self.fc_ratio(x)
        return cls_score, (bbox_pred, score_pred, ratio_pred)

    def get_targets_single(self, proposal, target, sampling_result):
        if sampling_result is None:
            gt_bbox, gt_bbox_ignore, gt_label, img_shape = self.get_targets_assign_parse(target)
            assign_result = self.assigner.assign(
                proposal, gt_bbox, gt_bbox_ignore, gt_label
            )
            sampling_result = self.sampler.sample(
                assign_result, proposal, gt_bbox, gt_label
            )

        if self.target_type is not None:
            hboxes, rboxes = self.get_targets_gt_parse(target)
            hboxes = hboxes[sampling_result.pos_assigned_gt_inds]
            rboxes = rboxes[sampling_result.pos_assigned_gt_inds]
        else:
            raise NotImplementedError

        num_pos = sampling_result.pos_bboxes.shape[0]
        num_neg = sampling_result.neg_bboxes.shape[0]
        num_samples = num_pos + num_neg
        labels = jt.zeros(num_samples, dtype=jt.int32)
        label_weights = jt.zeros(num_samples)
        bbox_targets = jt.zeros((num_samples, self.reg_dim))
        bbox_weights = jt.zeros((num_samples, self.reg_dim))
        score_targets = jt.zeros((num_samples, self.score_dim))
        score_weights = jt.zeros((num_samples, self.score_dim))
        ratio_targets = jt.zeros((num_samples, self.ratio_dim))
        ratio_weights = jt.zeros((num_samples, self.ratio_dim))

        if num_pos > 0:
            labels[:num_pos] = sampling_result.pos_gt_labels
            pos_weight = 1.0 if self.pos_weight < 0 else self.pos_weight
            label_weights[:num_pos] = pos_weight

            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, hboxes)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1.0

            pos_ratio_targets, pos_score_targets = self.cobb_coder.encode(rboxes)
            ratio_targets[:num_pos, :] = pos_ratio_targets
            ratio_weights[:num_pos, :] = 1.0
            score_targets[:num_pos, :] = pos_score_targets
            score_weights[:num_pos, :] = 1.0

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, score_targets, ratio_targets, \
                    bbox_weights, score_weights, ratio_weights, sampling_result

    def get_targets(self, proposals, targets, sampling_results=None, concat=True):
        if sampling_results is None:
            sampling_results = [None] * len(targets)
        (labels, label_weights, bbox_targets, score_targets, ratio_targets, bbox_weights, score_weights, ratio_weights,\
            sampling_result) = multi_apply(self.get_targets_single, proposals, targets, sampling_results)

        if concat:
            labels = jt.concat(labels, 0)
            label_weights = jt.concat(label_weights, 0)
            bbox_targets = jt.concat(bbox_targets, 0)
            score_targets = jt.concat(score_targets, 0)
            ratio_targets = jt.concat(ratio_targets, 0)
            bbox_weights = jt.concat(bbox_weights, 0)
            score_weights = jt.concat(score_weights, 0)
            ratio_weights = jt.concat(ratio_weights, 0)

        return labels, label_weights, (bbox_targets, score_targets, ratio_targets),\
            (bbox_weights, score_weights, ratio_weights), sampling_result

    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, 
             bbox_targets, bbox_weights, sampling_results):
        bbox_pred, score_pred, ratio_pred = bbox_pred
        bbox_targets, score_targets, ratio_targets = bbox_targets
        bbox_weights, score_weights, ratio_weights = bbox_weights
        if self.with_cls:
            loss_cls = self.loss_cls(cls_score, labels, label_weights)
        if self.with_reg:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.reshape(-1, self.reg_dim)[pos_inds]
                pos_score_pred = score_pred.reshape(-1, self.score_dim)[pos_inds]
                pos_ratio_pred = ratio_pred.reshape(-1, self.ratio_dim)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.reshape(-1, self.num_classes, self.reg_dim)[pos_inds, labels[pos_inds]]
                pos_score_pred = score_pred.reshape(-1, self.num_classes, self.score_dim)[pos_inds, labels[pos_inds]]
                pos_ratio_pred = ratio_pred.reshape(-1, self.num_classes, self.ratio_dim)[pos_inds, labels[pos_inds]]
            loss_bbox = self.loss_bbox(pos_bbox_pred,
                                       bbox_targets[pos_inds],
                                       bbox_weights[pos_inds],
                                       avg_factor=bbox_targets.shape[0])
            loss_score = self.loss_score(pos_score_pred,
                                     score_targets[pos_inds],
                                     score_weights[pos_inds],
                                     avg_factor=score_targets.shape[0])
            loss_ratio = self.loss_ratio(pos_ratio_pred,
                                         ratio_targets[pos_inds],
                                         ratio_weights[pos_inds],
                                         avg_factor=ratio_targets.shape[0])
        return dict(cobb_loss_cls=loss_cls, cobb_loss_bbox=loss_bbox,
                    cobb_loss_score=loss_score, cobb_loss_ratio=loss_ratio)

    def get_det_bboxes_single(self, proposals, cls_score, bbox_pred, target):
        bbox_pred, score_pred, ratio_pred = bbox_pred
        cfg = self.cfg

        if not self.reg_class_agnostic:
            bbox_pred = jt.reshape(bbox_pred, (bbox_pred.shape[0], -1, self.reg_dim))
            proposals = jt.expand(proposals[:, None, :], bbox_pred.shape)
            bbox_pred = jt.reshape(bbox_pred, (-1, self.reg_dim))
            proposals = jt.reshape(proposals, (-1, self.reg_dim))
            score_pred = jt.reshape(score_pred, (-1, self.score_dim))
        bboxes = self.bbox_coder.decode(proposals, bbox_pred, target['img_size'][::-1])
        rbboxes = self.cobb_coder.decode(bboxes, ratio_pred, score_pred)

        if not self.reg_class_agnostic:
            rbboxes = jt.reshape(rbboxes, (-1, self.num_classes * 5))

        scores = nn.softmax(cls_score, dim=-1) if cls_score is not None else None
        if cfg.rescale:
            rbboxes[:, 0::5] /= target['scale_factor']
            rbboxes[:, 1::5] /= target['scale_factor']
            rbboxes[:, 2::5] /= target['scale_factor']
            rbboxes[:, 3::5] /= target['scale_factor']
        det_bboxes, det_labels = multiclass_nms_rotated(rbboxes, scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        
        boxes = det_bboxes[:, :5]
        scores = det_bboxes[:, 5]
        polys = rotated_box_to_poly(boxes)
        return polys, scores, det_labels

    def get_det_bboxes(self, rois, cls_scores, bbox_preds, targets):
        bbox_preds, score_preds, ratio_preds = bbox_preds
        img_idx = rois[:, 0]
        results = []
        for i, target in enumerate(targets):
            cls_score = cls_scores[img_idx == i]
            bbox_pred = bbox_preds[img_idx == i]
            score_pred = score_preds[img_idx == i]
            ratio_pred = ratio_preds[img_idx == i]
            proposals = rois[img_idx == i, 1:]
            results.append(self.get_det_bboxes_single(proposals, cls_score, (bbox_pred, score_pred, ratio_pred), target))
        return results

    def get_refine_proposals_single(self, proposals, bbox_pred, target, label=None, sampling_result=None, filter_gt=False):
        bbox_pred, score_pred, ratio_pred = bbox_pred
        if not self.reg_class_agnostic:
            raise NotImplementedError
        assert bbox_pred.shape[1] == self.reg_dim
        assert score_pred.shape[1] == self.score_dim
        assert ratio_pred.shape[1] == self.ratio_dim

        bboxes = self.bbox_coder.decode(proposals, bbox_pred, target['img_size'][::-1])
        bbox_pred = self.cobb_coder.decode(bboxes, ratio_pred, score_pred)

        if filter_gt:
            assert sampling_result is not None
            num_rois = bbox_pred.shape[0]
            pos_keep = 1 - sampling_result.pos_is_gt
            keep = jt.ones((num_rois), dtype=jt.bool)
            keep[:len(sampling_result.pos_is_gt)] = pos_keep
            return bbox_pred[keep]
        else:
            return bbox_pred
    
    def get_refine_proposals(self, rois, bbox_preds, targets, labels=None, sampling_results=None, filter_gt=False):
        bbox_preds, score_preds, ratio_preds = bbox_preds

        img_idx = rois[:, 0]
        results = []
        for i, target in enumerate(targets):
            keep_inds = img_idx == i
            label = labels[keep_inds] if labels is not None else None
            bbox_pred, score_pred, ratio_pred = bbox_preds[keep_inds], score_preds[keep_inds], ratio_preds[keep_inds]
            bbox_pred = (bbox_pred, score_pred, ratio_pred)
            proposals = rois[keep_inds, 1:]
            sampling_result = sampling_results[i] if sampling_results is not None else None
            results.append(self.get_refine_proposals_single(proposals, bbox_pred, target, label, sampling_result, filter_gt))

        return results

@HEADS.register_module()
class SharedCOBBRoIHead(COBBRoIHead):
    def __init__(self, *args, num_fcs=2, fc_out_channels=1024, **kwargs):
        assert num_fcs >= 1
        super(SharedCOBBRoIHead, self).__init__(
            *args,
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            **kwargs)

@HEADS.register_module()
class COBBRoIHeadRbboxes(COBBRoIHead):
    def get_targets_single(self, proposal, target, sampling_result):
        if sampling_result is None:
            gt_bbox, gt_bbox_ignore, gt_label, img_shape = self.get_targets_assign_parse(target)
            assign_result = self.assigner.assign(
                proposal, gt_bbox, gt_bbox_ignore, gt_label
            )
            sampling_result = self.sampler.sample(
                assign_result, proposal, gt_bbox, gt_label
            )

        if self.target_type is not None:
            # ignore hboxes
            hboxes, rboxes = self.get_targets_gt_parse(target)
            hboxes = hboxes[sampling_result.pos_assigned_gt_inds]
            rboxes = rboxes[sampling_result.pos_assigned_gt_inds]
        else:
            raise NotImplementedError

        num_pos = sampling_result.pos_bboxes.shape[0]
        num_neg = sampling_result.neg_bboxes.shape[0]
        num_samples = num_pos + num_neg
        labels = jt.zeros(num_samples, dtype=jt.int32)
        label_weights = jt.zeros(num_samples)
        bbox_targets = jt.zeros((num_samples, self.reg_dim))
        bbox_weights = jt.zeros((num_samples, self.reg_dim))
        score_targets = jt.zeros((num_samples, self.score_dim))
        score_weights = jt.zeros((num_samples, self.score_dim))
        ratio_targets = jt.zeros((num_samples, self.ratio_dim))
        ratio_weights = jt.zeros((num_samples, self.ratio_dim))

        if num_pos > 0:
            labels[:num_pos] = sampling_result.pos_gt_labels
            pos_weight = 1.0 if self.pos_weight < 0 else self.pos_weight
            label_weights[:num_pos] = pos_weight

            pos_bbox_targets, pos_ratio_targets, pos_score_targets = \
                self.bbox_coder.encode(sampling_result.pos_bboxes, rboxes)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1.0

            ratio_targets[:num_pos, :] = pos_ratio_targets
            ratio_weights[:num_pos, :] = 1.0
            score_targets[:num_pos, :] = pos_score_targets
            score_weights[:num_pos, :] = 1.0

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, score_targets, ratio_targets, \
                    bbox_weights, score_weights, ratio_weights, sampling_result

    def get_det_bboxes_single(self, proposals, cls_score, bbox_pred, target):
        bbox_pred, score_pred, ratio_pred = bbox_pred
        cfg = self.cfg

        if not self.reg_class_agnostic:
            bbox_pred = jt.reshape(bbox_pred, (bbox_pred.shape[0], -1, self.reg_dim))
            proposals = jt.expand(proposals[:, None, :], bbox_pred.shape)
            bbox_pred = jt.reshape(bbox_pred, (-1, self.reg_dim))
            proposals = jt.reshape(proposals, (-1, self.reg_dim))
            score_pred = jt.reshape(score_pred, (-1, self.score_dim))
        rbboxes = self.bbox_coder.decode(proposals, bbox_pred, ratio_pred, score_pred)

        if not self.reg_class_agnostic:
            rbboxes = jt.reshape(rbboxes, (-1, self.num_classes * 5))

        scores = nn.softmax(cls_score, dim=-1) if cls_score is not None else None
        if cfg.rescale:
            rbboxes[:, 0::5] /= target['scale_factor']
            rbboxes[:, 1::5] /= target['scale_factor']
            rbboxes[:, 2::5] /= target['scale_factor']
            rbboxes[:, 3::5] /= target['scale_factor']
        det_bboxes, det_labels = multiclass_nms_rotated(rbboxes, scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        
        boxes = det_bboxes[:, :5]
        scores = det_bboxes[:, 5]
        polys = rotated_box_to_poly(boxes)
        return polys, scores, det_labels

    def get_refine_proposals_single(self, proposals, bbox_pred, target, label=None, sampling_result=None, filter_gt=False):
        bbox_pred, score_pred, ratio_pred = bbox_pred
        if not self.reg_class_agnostic:
            raise NotImplementedError
        assert bbox_pred.shape[1] == self.reg_dim
        assert score_pred.shape[1] == self.score_dim
        assert ratio_pred.shape[1] == self.ratio_dim

        bbox_pred = self.bbox_coder.decode(proposals, bbox_pred, score_pred, ratio_pred, score_pred)

        if filter_gt:
            assert sampling_result is not None
            num_rois = bbox_pred.shape[0]
            pos_keep = 1 - sampling_result.pos_is_gt
            keep = jt.ones((num_rois), dtype=jt.bool)
            keep[:len(sampling_result.pos_is_gt)] = pos_keep
            return bbox_pred[keep]
        else:
            return bbox_pred

@HEADS.register_module()
class SharedCOBBRoIHeadRbboxes(COBBRoIHeadRbboxes):
    def __init__(self, *args, num_fcs=2, fc_out_channels=1024, **kwargs):
        assert num_fcs >= 1
        super(SharedCOBBRoIHeadRbboxes, self).__init__(
            *args,
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            **kwargs)

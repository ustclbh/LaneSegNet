import os
import math
import torch
from mmdet.core import build_assigner
from .single_stage import SingleStageDetector
from ..builder import DETECTORS, build_loss
from ..necks.geb import GraphEnhancementModule


@DETECTORS.register_module
class LaneSegNet(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss='LaneLossAggress',
                 loss_weights={},
                 output_scale=4,
                 num_classes=1,
                 point_scale=True,
                 sample_gt_points=[11, 11, 11, 11],
                 assigner_cfg=dict(type='LaneAssigner'),
                 use_smooth=False,
                 use_geb=False,
                 geb_cfg=None):
        super(LaneSegNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=head,
            train_cfg=None,
            test_cfg=None,
            pretrained=pretrained)
        self.sample_gt_points = sample_gt_points
        self.num_classes = num_classes
        self.head = head
        self.use_smooth = use_smooth
        self.assigner_cfg = assigner_cfg
        self.loss_weights = loss_weights
        self.point_scale = point_scale
        self.use_geb = use_geb
        if test_cfg is not None and 'out_scale' in test_cfg.keys():
            self.output_scale = test_cfg['out_scale']
        else:
            self.output_scale = output_scale
        self.loss = build_loss(loss)
        if self.assigner_cfg:
            self.assigner = build_assigner(self.assigner_cfg)
        if self.use_geb and geb_cfg is not None:
            self.geb = GraphEnhancementModule(**geb_cfg)

    def forward(self, img, img_metas=None, return_loss=True, **kwargs):
        if img_metas is None:
            return self.test_inference(img, **kwargs)
        elif return_loss:
            return self.forward_train(img, img_metas, **kwargs)  # img shape [8 3 320 800]
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self, img, img_metas, **kwargs):
        # kwargs -> ['exist_mask', 'instance_mask', 'gauss_mask', 'hm_down_scale', 'lane_points']
        output = self.backbone(img.type(torch.cuda.FloatTensor))  # shape [8 128 80 200]  swin [B C 40 100]
        output = self.neck(output)  # features, deform_points
        
        # Apply GEB if enabled
        if self.use_geb and hasattr(self, 'geb'):
            geb_flag = True

        if self.head:
            [cpts_hm, kpts_hm, pts_offset, int_offset, rela_offset] = self.bbox_head.forward_train(output['features'],
                                                                                      output.get("aux_feat", None), self.geb)
        

        loss_items = [
            {"type": "focalloss", "gt": kwargs['gt_cpts_hm'], "pred": cpts_hm, "weight": self.loss_weights["center"]},
            {"type": "focalloss", "gt": kwargs['gt_kpts_hm'], "pred": kpts_hm, "weight": self.loss_weights["point"]},
            {"type": "smoothl1loss", "gt": kwargs['rela_offset_maps'], "pred": rela_offset,
             "weight": self.loss_weights["rela_offset"]}
            ]
        
        

        if not self.use_smooth:
            loss_items.append({"type": "regl1kploss", "gt": kwargs['int_offset'], "pred": int_offset,
                               "mask": kwargs['offset_mask'], "weight": self.loss_weights["error"]})
            loss_items.append({"type": "regl1kploss", "gt": kwargs['pts_offset'], "pred": pts_offset,
                               "mask": kwargs['offset_mask_weight'], "weight": self.loss_weights["offset"]})
        else:
            loss_items.append({"type": "smoothl1loss", "gt": kwargs['int_offset'], "pred": int_offset,
                               "mask": kwargs['offset_mask'], "weight": self.loss_weights["error"]})
            loss_items.append({"type": "smoothl1loss", "gt": kwargs['pts_offset'], "pred": pts_offset,
                               "mask": kwargs['offset_mask_weight'], "weight": self.loss_weights["offset"]})
        
        if "deform_points" in output.keys() and self.loss_weights["aux"] != 0:
            for i, points in enumerate(output['deform_points']):
                if points is None:
                    continue
                gt_points = kwargs[f'lane_points_l{i}']
                gt_matched_points, pred_matched_points = self.assigner.assign(points, gt_points,
                                                                              sample_gt_points=self.sample_gt_points[i])
                if self.point_scale:
                    loss_item = {"type": "smoothl1loss", "gt": gt_matched_points / (2 ** (3 - i)),
                                 "pred": pred_matched_points / (2 ** (3 - i)), "weight": self.loss_weights["aux"]}
                else:
                    loss_item = {"type": "smoothl1loss", "gt": gt_matched_points, "pred": pred_matched_points,
                                 "weight": self.loss_weights["aux"]}
                loss_items.append(loss_item)

        losses = self.loss(loss_items)
        return losses

    def test_inference(self, img, hack_seeds=None, **kwargs):
        """Test without augmentation."""
        import time
        import sys
        from mmdet.utils import get_model_complexity_info
        # # 获取当前日期和时间
        # current_datetime =  time.time()
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        # print("-------------------backbone:")
        def get_model_parameters_number(model):
            params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return params_num
        
        # print("-------------------backbone num param ----")
        # print(get_model_parameters_number(self.backbone)/1e6)
        # print("backbone",  time.time()-current_datetime)
        
        current_datetime = time.time()
        
        output = self.neck(output)  # shape [8 64 80 200]
        # print("-------------------neck num param ----")
        # print(get_model_parameters_number(self.neck)/1e6)
        # # print("neck",  time.time()-current_datetime)
        
        # Apply GEB if enabled
        if self.use_geb and hasattr(self, 'geb'):
            geb_flag = True

       
        current_datetime = time.time()
        if self.head:
            current_datetime = time.time()
            [cpts_hm, kpts_hm, pts_offset, int_offset, rela_offset] = self.bbox_head.forward_train(output['features'],
                                                                                      output.get("aux_feat", None), self.geb)
            seeds, hm = self.bbox_head.forward_test(output['features'], output.get("aux_feat", None), hack_seeds)
            # print("head",  time.time()-current_datetime)
            
            # print("-------------------head param ----")
            # print(get_model_parameters_number(self.bbox_head)/1e6)
            
            

        output['cpts_hm'] = cpts_hm
        output['kpts_hm'] = kpts_hm
        output['pts_offset'] = pts_offset
        output['int_offset'] = int_offset
        output['deform_points'] = output['deform_points']
        output['seeds'] = seeds
        output['hm'] = hm
        return output



    def forward_test(self, img, img_metas,
                     hack_seeds=None,
                     **kwargs):
        """Test without augmentation."""
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        output = self.neck(output)
        if self.head:
            seeds, hm = self.bbox_head.forward_test(output['features'], output.get("aux_feat", None), hack_seeds)
        return [seeds, hm]

    def forward_dummy(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        x = self.bbox_head.forward_train(x['features'], x.get("aux_feat", None))
        return x

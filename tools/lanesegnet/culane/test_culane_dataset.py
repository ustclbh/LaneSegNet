import argparse
import os

import mmcv
import cv2
import numpy as np
import torch
#import matplotlib.pyplot as plt
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils.general_utils import mkdir
from mmdet.models import build_detector
#from ..common import COLORS
from mmcv.parallel import MMDataParallel
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config', required=True,
                        help='test config file path')
    parser.add_argument('--show', default='./work_dirs/watch', required=True, help='show results')
    parser.add_argument('--max_show_num', type=int, default=10, help='show results')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file path')
    args = parser.parse_args()
    return args

def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros([h, w, 3], dtype=np.uint8)
    for i in range(np.max(mask)+1):
        rgb[mask == i] = COLORS[i]
    return rgb


def vis_one(data):
    # image
    img = data['img'].data[0].detach().cpu().numpy()[0, :, :, :]
    norm_cfg = data['img_metas'].data[0][0]['img_norm_cfg']
    downscale = data['img_metas'].data[0][0]['down_scale']
    hm_downscale = data['img_metas'].data[0][0]['hm_down_scale']
    height, width = data['img_metas'].data[0][0]['img_shape'][:2]
    img = img.transpose(1, 2, 0)
    img = (img * norm_cfg['std']) + norm_cfg['mean']
    img = img.astype(np.uint8)
    # hm
    print(data.keys())
    gt_hm = data['gt_kpts_hm'].data[0].detach().cpu().numpy()[
        0, :, :, :] * 255
    vis_hm = np.zeros_like(gt_hm[0])
    for i in range(gt_hm.shape[0]):
        vis_hm += gt_hm[i, :, :]

    gt_masks = data['img_metas'].data[0][0]['gt_masks']
    for i, mask_info in enumerate(gt_masks):
        vis_img = np.zeros(img.shape, np.uint8)
        vis_img[:] = img[:]
        line = []
        row = mask_info['row']
        vertical_range = mask_info['range']
        for idx, (coord_x, valid) in enumerate(zip(row, vertical_range[0])):
            point = (int(coord_x*downscale), int(idx * downscale))
            line.append(point)
        points = mask_info['points']
        label = mask_info['label']
        color = COLORS[label+1]
        for p in points:
            cv2.circle(vis_img, (hm_downscale*p[0], hm_downscale*p[1]), 3, color, -1)
            cv2.circle(vis_img, (hm_downscale*p[0], hm_downscale*p[1]), 1, (0,0,0), -1)
        if len(line) > 1:
            for pt1, pt2 in zip(line[:-1], line[1:]):
                cv2.line(vis_img, pt1, pt2, color, thickness=1)
        img = vis_img
    return img, vis_hm

def main():
    args = parse_args()
    mkdir(args.show)
    # build the dataloader
    cfg = mmcv.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data['workers_per_gpu'],
        dist=False,
        shuffle=False)
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['state_dict'], strict=True)
    # model.eval()
    model = MMDataParallel(model.cuda(), device_ids=[0])
    for index, data in enumerate(data_loader):
        file_name = data['img_metas'].data[0][0]['filename']
        save_name = os.path.splitext(os.path.basename(file_name))[0]
        print(index, file_name)
        # vis_img, vis_hm = vis_one(data)
        gt_kpts_hm, gt_cpts = vis_gt_hm(data)
        vis_img_dir = os.path.join(args.show, '{}_kpts_hm.png'.format(save_name))
        vis_hm_dir = os.path.join(args.show, '{}_cpts_hm.png'.format(save_name))
        #vis and save gt_kpts_hm
        cv2.imwrite(vis_img_dir, gt_kpts_hm[0])
        cv2.imwrite(vis_hm_dir, gt_cpts[0])

        #vis and save pts_offset
        pts_offset = data['pts_offset'].data[0].detach().cpu().numpy()[0]
        vis_pts_offset = pts_offset.transpose(1, 2, 0)
        #l2 norm of vis_pts_offset
        vis_pts_offset = np.linalg.norm(vis_pts_offset, axis=2)
        vis_pts_offset = (vis_pts_offset ).astype(np.uint8)
        vis_pts_offset_dir = os.path.join(args.show, '{}_pts_offset.png'.format(save_name))
        cv2.imwrite(vis_pts_offset_dir, vis_pts_offset)

        #vis and save int_offset
        int_offset = data['int_offset'].data[0].detach().cpu().numpy()[0]
        vis_int_offset = int_offset.transpose(1, 2, 0)
        #l2 norm of vis_int_offset
        vis_int_offset = np.linalg.norm(vis_int_offset, axis=2)
        vis_int_offset = (vis_int_offset *255).astype(np.uint8)
        vis_int_offset_dir = os.path.join(args.show, '{}_int_offset.png'.format(save_name))
        cv2.imwrite(vis_int_offset_dir, vis_int_offset)

        # results['deform_points'][0][0,:,20,50]
        #tensor([ 0.5694, 12.8553,  0.2401, -1.1791,  0.2473, -1.3239,  0.3106,  1.5232,
        #   0.4705,  4.4832, -0.3733, -4.3863,  0.1050,  8.5491], device='cuda:0')

        

        image   = data['img'].data[0].cuda()
        thr     = 0.3
        kpt_thr = 0.3
        cpt_thr = 0.3
        b = image.shape[0]
        results = model.module.test_inference(image, thr=thr, kpt_thr=kpt_thr, cpt_thr=cpt_thr)
        print(results.keys()) 
        #results.keys() => dict_keys(['features', 'aux_feat', 'deform_points', 'cpts_hm', 'kpts_hm',
        #  'pts_offset', 'int_offset', 'seeds', 'hm'])
        #features: b, 64, 40, 100  /20 50 /10 25
        #aux_feat: b, 64, 40, 100
        #deform_points: b, 14, 40, 100
        #cpts_hm: b, 1, 40, 100
        #kpts_hm: b, 1, 40, 100
        #pts_offset: b, 2, 40, 100
   


        if index >= args.max_show_num:
            break

def vis_gt_hm(data):
    gt_kpts_hm = data['gt_kpts_hm'].data[0].detach().cpu().numpy()[
        0, :, :, :] * 255
    gt_cpts = data['gt_cpts_hm'].data[0].detach().cpu().numpy()[0, :, :]* 255
    return gt_kpts_hm, gt_cpts

if __name__ == '__main__':
    main()
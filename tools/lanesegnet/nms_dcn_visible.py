#lbh_delete
import os
import PIL.Image
import PIL.ImageDraw
import argparse
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
#import scipy.interpolate as spi
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from post_process import PostProcessor
def adjust_result(lanes, centers, crop_bbox, img_shape, points_thr):
    h_img, w_img = img_shape[:2]
    ratio_x = (crop_bbox[2] - crop_bbox[0]) / w_img
    ratio_y = (crop_bbox[3] - crop_bbox[1]) / h_img
    offset_x, offset_y = crop_bbox[:2]

    results = []
    virtual_centers = []
    cluster_centers = []
    if lanes is not None:
        for key in range(len(lanes)):
            pts = []
            cts = []
            for pt in lanes[key]['points']:
                pt[0] = int(pt[0] * ratio_x + offset_x)
                pt[1] = int(pt[1] * ratio_y + offset_y)
                pts.append(tuple(pt))
            for ct in lanes[key]['centers']:
                ct[0] = int(ct[0] * ratio_x + offset_x)
                ct[1] = int(ct[1] * ratio_y + offset_y)
                cts.append(tuple(ct))
            # print('lane {} ====== \npoint nums {}'.format(key, len(pts)))
            # print('lane {} ====== \n point coord {}  \nvirtual center coord {}'.format(key, pts, cts))
            if len(pts) > points_thr:
                results.append(pts)
                virtual_centers.append(cts)
        # print('lane number:{}  virtual center number:{}'.format(len(results), len(virtual_centers)))
    if centers is not None:
        for center in centers:
            center_coord = center['center']
            center_coord[0] = int(center_coord[0] * ratio_x + offset_x)
            center_coord[1] = int(center_coord[1] * ratio_y + offset_y)
            cluster_centers.append(tuple(center_coord))

    return results, virtual_centers, cluster_centers


max_show_num = 10
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

def generate_grid(points_map):
    b, p, h, w = points_map.shape
    y = torch.arange(h)[:, None, None].repeat(1, w, 1)
    x = torch.arange(w)[None, :, None].repeat(h, 1, 1)
    # b, h, w, p
    coods = torch.cat([y, x], dim=-1)[None, :, :, None, :].repeat(b, 1, 1, p//2, 1).float()
    # b, p, h, w
    grid = coods.reshape(b, h, w, p).permute(0, 3, 1, 2).to(points_map.device)
    return grid
def path_join(path1, path2):
    return os.path.join(path1, path2)

def vis_one(results, virtual_center, cluster_center, filename, img_info=None, lane_width=7):
    def parse_img_info(img_info):
        converted_lanes = []
        h_samples = img_info['h_samples']
        lanes = img_info['lanes']
        for lane in lanes:
            converted_lane = []
            for coord_x, coord_y in zip(lane, h_samples):
                if coord_x >= 0:
                    converted_lane.append((coord_x, coord_y))
            converted_lanes.append(converted_lane)
        return converted_lanes

    img = cv2.imread(filename)
    img_gt = cv2.imread(filename)
    img_vc = cv2.imread(filename)
    img_circle = cv2.imread(filename)
    img_pil = PIL.Image.fromarray(img)
    img_gt_pil = PIL.Image.fromarray(img_gt)
    # img_vc_pil = PIL.Image.fromarray(img_vc)

    for idx, lane in enumerate(results):
        lane_tuple = [tuple(p) for p in lane]
        # print('lane tuple:', lane_tuple)
        PIL.ImageDraw.Draw(img_pil).line(
            xy=lane_tuple, fill=COLORS[idx + 1], width=lane_width)

    for idx, vp in enumerate(virtual_center):
        vp_tuple = [tuple(p) for p in vp]
        # print('vp tuple:', vp_tuple)
        # for _vp in vp_tuple:
        #     cv2.circle(img=img_vc, center=_vp, radius=3, color=COLORS[idx + 1], thickness=-1)
        for _vp in vp_tuple:
            cv2.circle(img=img_circle, center=_vp, radius=3, color=COLORS[idx + 1], thickness=-1)

    for idx, cp in enumerate(cluster_center):
        print(f"{cp}###############################################################")
        cv2.circle(img=img_vc, center=cp, radius=10, color=COLORS[idx + 1], thickness=-1)
        cv2.circle(img=img_circle, center=cp, radius=40, color=COLORS[idx + 1], thickness=3)


    img = np.array(img_pil, dtype=np.uint8)
    # img_vc = np.array(img_vc_pil, dtype=np.uint8)

    if img_info is not None:
        gt_lanes = parse_img_info(img_info)
        for idx, lane in enumerate(gt_lanes):
            lane_tuple = [tuple(p) for p in lane]
            PIL.ImageDraw.Draw(img_gt_pil).line(
                xy=lane_tuple, fill=COLORS[idx + 1], width=lane_width)
        img_gt = np.array(img_gt_pil, dtype=np.uint8)

    return img, img_gt, img_vc, img_circle



parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
parser.add_argument('--config_name', default="./configs/culane/final_exp_res18_s8.py")
args = parser.parse_args()

#config = f"configs/magiclanenet/tusimple/{args.config_name}.py"
config = './configs/culane/final_exp_res18_s8.py'
cfg = Config.fromfile(config)
cfg.data.samples_per_gpu = 1
gt_path = '/media/ssdspace/lbh/DATASET/tusimple/test_label.json'
# pred_path = sorted(glob.glob(f'tools/output/tusimple/{args.config_name}_2021*/result/test.json'))[-1]
# print(pred_path)
# # p: 多检测的 n: 少检测的
# criterias, _ = LaneEval.bench_one_submit(pred_path, gt_path, return_each=True)
# bad_p = torch.arange(len(criterias['p']))[torch.tensor(criterias['p']) > 0]
# bad_n = torch.arange(len(criterias['n']))[torch.tensor(criterias['n']) > 0]
# print(bad_p)
# print(bad_n)
# cfg.val_pipeline[-1] = cfg.train_pipeline[-1]
# cfg.data.train.pipeline = cfg.val_pipeline
dataset = build_dataset(cfg.data.train)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False)
#checkpoint = sorted(glob.glob(f'tools/output/tusimple/{args.config_name}_2021*/latest.pth'))[-1]
checkpoint = ''
print(checkpoint)
model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
model.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'], strict=True)
model.eval()
model = MMDataParallel(model.cuda(), device_ids=[0])

for i, data in enumerate(data_loader):
    label = 'None'
    # if i not in bad_p and i not in bad_n:
    #     continue
    # label = ''
    # if i in bad_p:
    #     label = label+'bad_p_'
    # if i in bad_n:
    #     label = label+'bad_n_'

    image   = data['img'].data[0].cuda()
    thr     = 0.3
    kpt_thr = 0.3
    cpt_thr = 0.3
    b = image.shape[0]
    results = model.module.test_inference(image, thr=thr, kpt_thr=kpt_thr, cpt_thr=cpt_thr)

    if not os.path.exists(f"debug/{args.config_name}/"):
        os.makedirs(f"debug/{args.config_name}/")
    # result map
    cpts_hm    = results['cpts_hm'][0, 0].detach().cpu().sigmoid().numpy()
    gt_cpts_hm = data['gt_cpts_hm'].data[0][0, 0].detach().cpu().numpy()
    kpts_hm    = results['kpts_hm'][0, 0].detach().cpu().sigmoid().numpy()
    gt_kpts_hm = data['gt_kpts_hm'].data[0][0, 0].detach().cpu().numpy()
    # seeds      = results['seeds'] # align
    # hm         = results['hm']
    image      = image[0, 0].detach().cpu().numpy()
    if not os.path.exists(f"debug/{args.config_name}/%04d/"%i):
        os.makedirs(f"debug/{args.config_name}/%04d/"%i)
    plt.imsave(f"debug/{args.config_name}/%04d/{label}result_cpts_hm.png"%i, cpts_hm)
    plt.imsave(f"debug/{args.config_name}/%04d/{label}gt_cpts_hm.png"%i,     gt_cpts_hm)
    plt.imsave(f"debug/{args.config_name}/%04d/{label}result_kpts_hm.png"%i, kpts_hm)
    plt.imsave(f"debug/{args.config_name}/%04d/{label}gt_kpts_hm.png"%i,     gt_kpts_hm)
    plt.imsave(f"debug/{args.config_name}/%04d/{label}image.png"%i,          image)


    #visualize dcn points
    layer = 0
    dp_num = cfg.dcn_point_num
    if not os.path.exists(f"debug/{args.config_name}/%04d/dcn_points_l{layer}/"%i):
        os.makedirs(f"debug/{args.config_name}/%04d/dcn_points_l{layer}/"%i)
    # cood = data[f'lane_points_l{layer}'].data[0][0].reshape(-1, 2).long()
    # print(cood)
    # cood = torch.cat([c[None, ...] for c in cood if c[0] > 0])
    # # cood = cood[cood>0]
    # y_ = cood[:, 0]
    # x_ = cood[:, 1]
    deform_point = results['deform_points'][layer]
    # exist_mask = torch.zeros(deform_point.shape[-2:])
    # exist_mask[y_, x_] = 1
    exist_mask = (data['gt_kpts_hm'].data[0][0, 0] > 0.5).long()
    # b, p, h, w
    grid = generate_grid(deform_point)
    pos_abs = deform_point+grid
    grid_filter = grid.contiguous()[:, :, exist_mask.bool()].reshape(b, dp_num[layer], 2, -1)
    pos_abs_filter = pos_abs.contiguous()[:, :, exist_mask.bool()].reshape(b, dp_num[layer], 2, -1)
    print(deform_point.shape)
    plt.figure(figsize=(100, 40))
    gaps = [16, 8, 4, 2]
    gap = gaps[layer]
    print(pos_abs_filter.shape)
    for p_id in range(0, pos_abs_filter.shape[-1], gap):
        c_p = grid_filter[0, 0, :, p_id].long()
        mask = exist_mask.numpy().copy()
        ym, xm = exist_mask.shape
        for p in pos_abs_filter[0, :, :, p_id].long():
            y = torch.clamp(p[0], 0, ym-1).cpu()
            x = torch.clamp(p[1], 0, xm-1).cpu()
            mask[y, x] = 2
        mask[c_p[0], c_p[1]] = 4
        plt.subplot(10, 4, p_id//gap+1)
        mask = np.uint8(mask/4*255)
        heat_img = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
        # plt.imshow(heat_img)
        heat_img = cv2.resize(heat_img, (400, 160), interpolation=cv2.INTER_NEAREST)
        plt.imsave(f"debug/{args.config_name}/%04d/dcn_points_l{layer}/%03d.png"%(i,p_id), heat_img)
    
    hm_thr=cfg.hm_thr
    kpt_thr=cfg.kpt_thr
    cpt_thr=cfg.cpt_thr
    points_thr=cfg.points_thr
    result_dst=f"debug/{args.config_name}/%04d"%i
    cluster_thr=cfg.cluster_thr
    crop_bbox=(0, 160, 1280, 720)
    sub_name = data['img_metas'].data[0][0]['sub_img_name']
    downscale = data['img_metas'].data[0][0]['hm_down_scale']
    
    img_shape = data['img_metas'].data[0][0]['img_shape']
    ori_shape = data['img_metas'].data[0][0]['ori_shape']
    

    post_processor = PostProcessor(use_offset=True, cluster_thr=cluster_thr, group_fast=cfg.group_fast, cluster_by_center_thr=cfg.cluster_by_center_thr)
    lanes, cluster_centers = post_processor([results['seeds'], results['hm']], downscale)
    result, virtual_center, cluster_center = adjust_result(
        lanes=lanes, centers=cluster_centers, crop_bbox=crop_bbox,
        img_shape=img_shape, points_thr=points_thr)
    filename = data['img_metas'].data[0][0]['filename']
    # img_info = data['img_metas'].data[0][0]['img_info']
    img_vis, img_gt_vis, virtual_center_vis, img_circle = vis_one(result, virtual_center, cluster_center, filename, None)
    save_name = sub_name.replace('/', '.')
    dst_show_dir = path_join(result_dst, save_name)
    dst_show_gt_dir = path_join(result_dst, save_name + '.gt.jpg')
    dst_show_vc_dir = path_join(result_dst, save_name + '.vc.jpg')
    dst_show_img_dir = path_join(result_dst, save_name + '.img.jpg')
    cv2.imwrite(dst_show_dir, img_vis)
    cv2.imwrite(dst_show_gt_dir, img_gt_vis)
    cv2.imwrite(dst_show_vc_dir, virtual_center_vis)
    cv2.imwrite(dst_show_img_dir, cv2.imread(filename))
    if i >= max_show_num:
        break



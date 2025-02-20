import numpy as np
import torch
from torch import nn
import torch.functional as F
import torch.nn.functional as F

from ..builder import HEADS
from .ctnet_head import CtnetHead


def compute_locations(shape, device):
    pos = torch.arange(
        0, shape[-1], step=1, dtype=torch.float32, device=device)
    pos = pos.reshape((1, 1, -1))
    pos = pos.repeat(shape[0], shape[1], 1)
    return pos


def make_mask(shape=(1, 80, 200), device=torch.device('cuda')):
    x_coord = torch.arange(0, shape[-1], step=1, dtype=torch.float32, device=device)
    x_coord = x_coord.reshape(1, 1, -1)
    # x_coord = np.repeat(x_coord, shape[1], 1)
    x_coord = x_coord.repeat(1, shape[1], 1)
    y_coord = torch.arange(0, shape[-2], step=1, dtype=torch.float32, device=device)
    y_coord = y_coord.reshape(1, -1, 1)
    y_coord = y_coord.repeat(1, 1, shape[-1])
    coord_mat = torch.cat((x_coord, y_coord), axis=0)
    # print('coord_mat shape{}'.format(coord_mat.shape))
    return coord_mat


def make_coordmat(shape=(1, 80, 200), device=torch.device('cuda')):
    x_coord = torch.arange(0, shape[-1], step=1, dtype=torch.float32, device=device)
    x_coord = x_coord.reshape(1, 1, -1)
    # x_coord = np.repeat(x_coord, shape[1], 1)
    x_coord = x_coord.repeat(1, shape[1], 1)
    y_coord = torch.arange(0, shape[-2], step=1, dtype=torch.float32, device=device)
    y_coord = y_coord.reshape(1, -1, 1)
    y_coord = y_coord.repeat(1, 1, shape[-1])
    coord_mat = torch.cat((x_coord, y_coord), axis=0)
    # print('coord_mat shape{}'.format(coord_mat.shape))
    return coord_mat


class UpSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        # self.upsample = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1,
        #                        output_padding=1),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU()
        # )

    def forward(self, x):
        out = self.Conv_BN_ReLU_2(x)
        # out = self.upsample(out)
        out = F.interpolate(input=out, scale_factor=2, mode='bilinear')
        return out


@HEADS.register_module()
class HeadFast(nn.Module):

    def __init__(self,
                 heads,
                 in_channels,
                 num_classes,
                 branch_in_channels=288,
                 hm_idx=0,  # input id for heatmap
                 joint_nums=1,
                 regression=True,
                 upsample_num=0,
                 root_thr=1,
                 train_cfg=None,
                 test_cfg=None):
        super(HeadFast, self).__init__()
        self.root_thr = root_thr
        self.num_classes = num_classes
        self.hm_idx = hm_idx
        self.joint_nums = joint_nums
        if upsample_num > 0:
            self.upsample_module = nn.ModuleList([UpSampleLayer(in_ch=branch_in_channels, out_ch=branch_in_channels)
                                                  for i in range(upsample_num)])
        else:
            self.upsample_module = None

        self.centerpts_head = CtnetHead(
            heads,   #heads=dict(hm=num_lane_classes),
            channels_in=branch_in_channels,
            final_kernel=1,
            head_conv=branch_in_channels)

        self.keypts_head = CtnetHead(
            heads,
            channels_in=branch_in_channels,
            final_kernel=1,
            head_conv=branch_in_channels)

        self.offset_head = CtnetHead(
            heads=dict(offset_map=self.joint_nums * 2),
            channels_in=branch_in_channels,
            final_kernel=1,
            head_conv=branch_in_channels)
        
        self.rela_offset_head = CtnetHead(
            heads=dict(offset_map=40),
            channels_in=branch_in_channels,
            final_kernel=1,
            head_conv=branch_in_channels)

        self.reg_head = CtnetHead(
            heads=dict(offset_map=2),
            channels_in=branch_in_channels,
            final_kernel=1,
            head_conv=branch_in_channels)

    def ktdet_decode(self, heat, offset, error, thr=0.1):

        def _nms(heat, kernel=3):
            pad = (kernel - 1) // 2
            hmax = nn.functional.max_pool2d(heat, (1, 3), stride=(1, 1), padding=(0, 1))
            keep = (hmax == heat).float()  # false:0 true:1
            return heat * keep  # type: tensor

        def check_range(start, end, value):
            if value < start:
                # print('out range value:{}'.format(value))
                return start
            elif value > end:
                # print('out range value:{}'.format(value))
                return end
            else:
                return value

        def get_virtual_down_coord(coord, offset_map, root_i):
            x, y = coord[0], coord[1]
            x_max = offset_map.shape[1] - 1
            y_max = offset_map.shape[0] - 1
            x = check_range(0, x_max, value=x)
            y = check_range(0, y_max, value=y)
            offset_vector = offset_map[y, x]
            offset_vector = offset_vector.reshape(-1, 2)
            offset_min_idx, offset_min_value = 0, 9999
            for idx, _offset in enumerate(offset_vector):
                offset_y = _offset[1]
                if offset_y < 0:
                    continue
                if offset_y < offset_min_value:
                    offset_min_value = offset_y
                    offset_min_idx = idx
            if offset_min_value < 5 and offset_min_idx > 0:
                offset_min_idx = offset_min_idx - 1
            offset_min = offset_vector[offset_min_idx]
            virtual_down_x, virtual_down_y = x + offset_min[0] + 0.49999, y + offset_min[1] + 0.49999
            virtual_down_coord = [int(virtual_down_x), int(virtual_down_y)]
            return virtual_down_coord

        def get_vitual_root(coord, offset_map):
            virtual_down_root0 = get_virtual_down_coord(coord, offset_map, 0)
            virtual_down_root1 = get_virtual_down_coord(virtual_down_root0, offset_map, 1)
            virtual_down_root2 = get_virtual_down_coord(virtual_down_root1, offset_map, 2)
            virtual_down_root3 = get_virtual_down_coord(virtual_down_root2, offset_map, 3)
            return virtual_down_root3

        def get_vitual_root_one(coord, offset_map):
            virtual_down_root = get_virtual_down_coord(coord, offset_map, 0)
            return virtual_down_root

        def _format(heat, offset, error, inds):
            ret = []
            for y, x, c in zip(inds[0], inds[1], inds[2]):
                id_class = c + 1
                coord = [x, y]
                score = heat[y, x, c]
                if offset.shape[-1] == 2:
                    _virtual_root = get_vitual_root_one(coord, offset)
                else:
                    _virtual_root = get_vitual_root(coord, offset)
                _error = error[y, x]
                ret.append((np.int32(coord + _error), np.int32(_virtual_root)))
            return ret

        heat_nms = _nms(heat)
        heat_nms = heat_nms.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        offset = offset.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        error = error.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        inds = np.where(heat_nms > thr)
        seeds = _format(heat_nms, offset, error, inds)
        return seeds

    def ktdet_decode_fast(self, heat, offset, error, thr=0.1, root_thr=1):

        def _nms(heat, kernel=3):
            hmax = nn.functional.max_pool2d(heat, (1, 3), stride=(1, 1), padding=(0, 1))
            keep = (hmax == heat).float()  # false:0 true:1
            return heat * keep  # type: tensor

        heat_nms = _nms(heat)

        # generate root centers array from offset map parallel
        offset_split = torch.split(offset, 1, dim=1)
        mask = torch.lt(offset_split[1], root_thr)  # offset < 1
        mask_nms = torch.gt(heat_nms, thr)  # key point score > 0.3
        mask_low = mask * mask_nms
        mask_low = mask_low[0, 0].transpose(1, 0).detach().cpu().numpy()
        idx = np.where(mask_low)
        root_center_arr = np.array(idx, dtype=int).transpose()

        # generate roots by coord add offset parallel
        heat_nms = heat_nms.squeeze(0).permute(1, 2, 0).detach()
        offset = offset.squeeze(0).permute(1, 2, 0).detach()
        error = error.squeeze(0).permute(1, 2, 0).detach()
        coord_mat = make_coordmat(shape=heat.shape[1:])  # 0.2ms
        coord_mat = coord_mat.permute(1, 2, 0)
        # print('\nkpt thr:', thr)
        heat_mat = heat_nms.repeat(1, 1, 2)
        root_mat = coord_mat + offset
        align_mat = coord_mat + error
        inds_mat = torch.where(heat_mat > thr)
        root_arr = root_mat[inds_mat].reshape(-1, 2).cpu().numpy()
        align_arr = align_mat[inds_mat].reshape(-1, 2).cpu().numpy()
        kpt_seeds = []
        for (align, root) in (zip(align_arr, root_arr)):
            kpt_seeds.append((align, np.array(root, dtype=float)))

        return root_center_arr, kpt_seeds

    def forward_train(self, inputs, aux_feat=None, geb=None):
        x_list = list(inputs)
        f_hm = x_list[self.hm_idx]


        if self.upsample_module is not None:
            for upsample in self.upsample_module:
                f_hm = upsample(f_hm)
                if aux_feat is not None:
                    aux_feat = upsample(aux_feat)

        z = self.centerpts_head(f_hm)
        cpts_hm = z['hm']

        z_ = self.keypts_head(f_hm)
        kpts_hm = z_['hm']

        cpts_hm = torch.clamp(torch.sigmoid(cpts_hm), min=1e-4, max=1 - 1e-4)
        kpts_hm = torch.clamp(torch.sigmoid(kpts_hm), min=1e-4, max=1 - 1e-4)
        ref_mask, ref_pts_idx, flat_ref_nodes, gbl_adj_matrix, local_mask, loc_pts_idx, flat_lane_nodes, loc_adj_matrix =  self.preprocess(kpts_hm, f_hm)
        ref_nodes, flat_lane_nodes = geb(flat_ref_nodes, flat_lane_nodes, gbl_adj_matrix, loc_adj_matrix);

        
        N, C, H, W = f_hm.shape
        # add ref_nodes, flat_lane_nodes back to confidence_map according to ref_pts_idx, loc_pts_idx
        batch_idx = torch.arange(N, device=f_hm.device).unsqueeze(-1).unsqueeze(-1)
        ref_nodes = f_hm.view(N,C, -1)[batch_idx,:,ref_pts_idx.view(N,1, -1)].squeeze(1)
        ref_nodes += flat_ref_nodes 


        lane_nodes = f_hm.view(N,C, -1)[batch_idx,:,loc_pts_idx.view(N,1, -1)].squeeze(1)
        lane_nodes += flat_lane_nodes 



        if aux_feat is not None:
            f_hm = aux_feat
        o = self.offset_head(f_hm)
        pts_offset = o['offset_map']

        o_ = self.reg_head(f_hm)
        int_offset = o_['offset_map']

        o_rela = self.rela_offset_head(f_hm)
        rela_offset = o_rela['offset_map']

        return [cpts_hm, kpts_hm, pts_offset, int_offset, rela_offset]

    def forward_test(
            self,
            inputs,
            aux_feat=None,
            hack_seeds=None,
            geb=None,
            hm_thr=0.3,
            kpt_thr=0.4,
            cpt_thr=0.4,
    ):

        x_list = list(inputs)
        f_hm = x_list[self.hm_idx]
        if self.upsample_module is not None:
            for upsample in self.upsample_module:
                f_hm = upsample(f_hm)
                if aux_feat is not None:
                    aux_feat = upsample(aux_feat)
        # center points hm
        z = self.centerpts_head(f_hm)
        hm = z['hm']
        hm = torch.clamp(hm.sigmoid(), min=1e-4, max=1 - 1e-4)
        cpts_hm = hm

        # key points hm
        z_ = self.keypts_head(f_hm)
        kpts_hm = z_['hm']

        cpts_hm = torch.clamp(torch.sigmoid(cpts_hm), min=1e-4, max=1 - 1e-4)
        kpts_hm = torch.clamp(torch.sigmoid(kpts_hm), min=1e-4, max=1 - 1e-4)
        ref_mask, ref_pts_idx, flat_ref_nodes, gbl_adj_matrix, local_mask, loc_pts_idx, flat_lane_nodes, loc_adj_matrix =  self.preprocess(kpts_hm, f_hm)
        ref_nodes, flat_lane_nodes = geb(flat_ref_nodes, flat_lane_nodes, gbl_adj_matrix, loc_adj_matrix);

        N, C, H, W = f_hm.shape 
        f_hm.view(-1)[ref_pts_idx.view(-1)] += ref_nodes.view(-1)
        f_hm.view(-1)[loc_pts_idx.view(-1)] += flat_lane_nodes.view(-1)
        f_hm = f_hm.view(N, C, H, W)


        # offset map
        if aux_feat is not None:
            f_hm = aux_feat
        o = self.offset_head(f_hm)
        pts_offset = o['offset_map']

        o_ = self.reg_head(f_hm)
        int_offset = o_['offset_map']

        o_rela = self.rela_offset_head(f_hm)
        rela_offset = o_rela['offset_map']

        if pts_offset.shape[1] > 2:
            def _nms(heat, kernel=3):
                hmax = nn.functional.max_pool2d(heat, (1, 3), stride=(1, 1), padding=(0, 1))
                keep = (hmax == heat).float()  # false:0 true:1
                return heat * keep  # type: tensor

            heat_nms = _nms(kpts_hm)
            offset_split = torch.split(pts_offset, 1, dim=1)
            mask = torch.lt(offset_split[1], self.root_thr)  # offset < 1
            mask_nms = torch.gt(heat_nms, kpt_thr)  # key point score > 0.3
            mask_low = mask * mask_nms
            mask_low = torch.squeeze(mask_low).permute(1, 0).detach().cpu().numpy()
            idx = np.where(mask_low)
            cpt_seeds = np.array(idx, dtype=int).transpose()
            kpt_seeds = self.ktdet_decode(kpts_hm, pts_offset, int_offset,
                                          thr=kpt_thr)  # key point position list[dict{} ]
        else:
            cpt_seeds, kpt_seeds = self.ktdet_decode_fast(kpts_hm, pts_offset, int_offset, thr=kpt_thr,
                                                          root_thr=self.root_thr)

        return [cpt_seeds, kpt_seeds, rela_offset]

    def preprocess(self, confidence_map, feature_map, conf_threshold=0.4, max_polling_size=7, L=40, M=40):
        device = confidence_map.device  # Get the device from input tensor
        
        assert confidence_map.ndim == 4 and feature_map.ndim == 4
        N, K, H, W = confidence_map.shape
        _, C, _, _ = feature_map.shape

        assert (confidence_map >= 0).all() and (confidence_map <= 1).all()
        init_confidence = confidence_map
        thresh_confidence = (confidence_map >= conf_threshold).float() * confidence_map

        local_mask = torch.zeros_like(confidence_map)
        row_confidence, _ = thresh_confidence.max(dim=3, keepdim=True)
        local_mask = (thresh_confidence == row_confidence).float()

        loc_pts_idx = local_mask.view(N, K, -1).nonzero(as_tuple=False)[:, [0,-1]]
        flat_lane_nodes = feature_map.view(N, C, -1)[loc_pts_idx[:,0],:,loc_pts_idx[:,1]].view(N,K,-1,C)

        ref_mask = torch.zeros_like(init_confidence)
        max_pool = F.max_pool2d(local_mask*init_confidence, kernel_size=max_polling_size, stride=1, padding=max_polling_size // 2)
        ref_mask = (init_confidence == max_pool).float()

        temp = ref_mask.view(N, K, -1).nonzero(as_tuple=False)
        ref_pts_idx = temp[:, 2]
        flat_ref_nodes = feature_map.view(N, C, -1)[:,:,ref_pts_idx].transpose(1, 2)
        ref_num = flat_ref_nodes.shape[1]
        gbl_adj_matrix = torch.ones(N, ref_num, ref_num, device=device)  # Ensure on correct device

        result = torch.zeros(N, K, M, device=device)  # Ensure on correct device
        for n in range(N):
            for k in range(K):
                group_0 = temp[(temp[:, 1] == 0) & (temp[:, 0] == n)][:, 2]
                group_1 = temp[(temp[:, 1] == 1) & (temp[:, 0] == n)][:, 2]
                group_2 = temp[(temp[:, 1] == 2) & (temp[:, 0] == n)][:, 2]
                group_3 = temp[(temp[:, 1] == 3) & (temp[:, 0] == n)][:, 2]
                _ = min(len(group_0), M)
                result[n, k, :_] = group_0[:_]
                _ = min(len(group_1), M)
                result[n, k, :_] = group_1[:_]
                _ = min(len(group_2), M)
                result[n, k, :_] = group_2[:_]
                _ = min(len(group_3), M)
                result[n, k, :_] = group_3[:_]
        
        ref_pts_idx = result
        loc_pts_idx = loc_pts_idx[:,1]
        loc_pts_idx = loc_pts_idx.view(N, K, -1)

        loc_adj_matrix = self.generate_loc_adj_matrix(ref_pts_idx, loc_pts_idx).to(device)  # Ensure on correct device

        return ref_mask, ref_pts_idx, flat_ref_nodes, gbl_adj_matrix, local_mask, loc_pts_idx, flat_lane_nodes, loc_adj_matrix

    def generate_loc_adj_matrix(self, ref_pts_idx, loc_pts_idx):
        N, K, M = ref_pts_idx.shape
        _, _, L = loc_pts_idx.shape
        loc_adj_matrix = torch.zeros(N, K, L, L)
        #TODO: optimize for faster loop
        for n in range(N):
            for k in range(K):
                # 获取当前批次和车道的 ref_pts_idx 和 loc_pts_idx
                current_ref_pts = ref_pts_idx[n, k]
                current_loc_pts = loc_pts_idx[n, k]

                # 找到非零的 ref_pts_idx 位置，即有效参考点的索引
                valid_ref_indices = (current_ref_pts != 0).nonzero(as_tuple=True)[0]

                # 如果没有有效参考点，跳过当前车道
                if valid_ref_indices.numel() == 0:
                    continue

                # 根据参考点索引划分 loc_pts_idx 为多个 lane segment
                segments = []
                start_idx = 0
                for i in range(len(valid_ref_indices)):
                    end_idx = (current_loc_pts > current_ref_pts[valid_ref_indices[i]]).nonzero(as_tuple=True)[0]
                    if end_idx.numel() == 0:
                        end_idx = L
                    else:
                        end_idx = end_idx[0].item()
                    segments.append(current_loc_pts[start_idx:end_idx])
                    start_idx = end_idx

                # 处理最后一个 segment
                if start_idx < L:
                    segments.append(current_loc_pts[start_idx:])

                # 为相邻的 lane segment 建立连接
                for i in range(len(segments) - 1):
                    for j in range(len(segments[i])):
                        for l in range(len(segments[i + 1])):
                            idx1 = (current_loc_pts == segments[i][j]).nonzero(as_tuple=True)[0].item()
                            idx2 = (current_loc_pts == segments[i + 1][l]).nonzero(as_tuple=True)[0].item()
                            loc_adj_matrix[n, k, idx1, idx2] = 1
                            loc_adj_matrix[n, k, idx2, idx1] = 1
        return loc_adj_matrix

    def inference_mask(self, pos):
        pass

    def forward(
            self,
            x_list,
            hm_thr=0.3,
            kpt_thr=0.4,
            cpt_thr=0.4,
    ):
        return self.forward_test(x_list, hm_thr, kpt_thr, cpt_thr)

    def init_weights(self):
        # ctnet_head will init weights during building
        pass

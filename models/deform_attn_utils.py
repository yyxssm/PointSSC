# modify from https://github.com/TuSimple/centerformer/blob/master/det3d/models/utils/transformer.py # noqa

import torch
from einops import rearrange
from mmcv.cnn.bricks.activation import GELU
from torch import einsum, nn
import numpy as np

from mmcv.ops import knn, grouping_operation
from models.deform_attn_multi_scale import MSDeformAttn


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y=None, **kwargs):
        if y is not None:
            return self.fn(self.norm(x), self.norm(y), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class FFN(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 n_heads=8,
                 dim_single_head=64,
                 dropout=0.0,
                 out_attention=False):
        super().__init__()
        inner_dim = dim_single_head * n_heads
        project_out = not (n_heads == 1 and dim_single_head == dim)

        self.n_heads = n_heads
        self.scale = dim_single_head**-0.5
        self.out_attention = out_attention

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out else nn.Identity())

    def forward(self, x):
        _, _, _, h = *x.shape, self.n_heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # [bs, n_head, npoint, c]

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # [bs, n_head, npoint, npoint]

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # [bs, n_head, npoint, c]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [bs, npoint, C]

        if self.out_attention:
            return self.to_out(out), attn
        else:
            return self.to_out(out)


class SelfAttentionKNN(nn.Module):
    """
    自创模型，还需验证准确性
    """

    def __init__(self,
                 dim,
                 n_heads=8,
                 dim_single_head=64,
                 dropout=0.0,
                 out_attention=False,
                 neighbor_num=8,
                 ):
        super().__init__()
        inner_dim = dim_single_head * n_heads
        project_out = not (n_heads == 1 and dim_single_head == dim)
        self.neighbor_num = neighbor_num

        self.n_heads = n_heads
        self.scale = dim_single_head**-0.5
        self.out_attention = out_attention

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out else nn.Identity())

    def forward(self, x, center_pos3d, **kwargs):
        _, _, _, h = *x.shape, self.n_heads
        
        # 基于KNN进行邻近点的索引
        knn_id = knn(self.neighbor_num, center_pos3d, center_pos3d)
        # TODO: 现在是基于KNN进行索引的，但是KNN索引不能超过100个点，后续考虑换成ball_query进行索引
        knn_x = grouping_operation(x.transpose(1, 2), knn_id).permute(0, 3, 2, 1)  # [bs, npoint, nsample, C]
        
        q = self.to_q(x)  # [bs, npoint, C]
        kv = self.to_kv(knn_x).chunk(2, dim=-1)  # [bs, npoint, nsample, C]
        q = rearrange(q, 'b n (h d) -> b h n d', h=h).unsqueeze(3)  # [bs, n_head, npoint, 1, c]
        k, v = map(lambda t: rearrange(t, 'b n s (h d) -> b h n s d', h=h), kv)  # [bs, n_head, npoint, nsample, c]

        dots = einsum('b h n i d, b h n j d -> b h n i j', q, k) * self.scale  # [bs, n_head, npoint, 1, nsample]

        attn = self.attend(dots)

        out = einsum('b h n i j, b h n j d -> b h n i d', attn, v)  # [bs, n_head, npoint, 1, c]
        out = out.squeeze(3)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if self.out_attention:
            return self.to_out(out), attn
        else:
            return self.to_out(out)


class DeformableCrossAttention(nn.Module):

    def __init__(
        self,
        dim_model=256,
        dim_single_head=64,
        dropout=0.3,
        n_levels=3,
        n_heads=6,
        n_points=9,
        out_sample_loc=False,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(
            dim_model,
            dim_single_head,
            n_levels,
            n_heads,
            n_points,
            out_sample_loc=out_sample_loc)
        self.dropout = nn.Dropout(dropout)
        self.out_sample_loc = out_sample_loc

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        src,
        query_pos=None,
        reference_points=None,
        src_spatial_shapes=None,
        level_start_index=None,
        src_padding_mask=None,
    ):
        # cross attention
        tgt2, sampling_locations = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        tgt = self.dropout(tgt2)

        if self.out_sample_loc:
            return tgt, sampling_locations
        else:
            return tgt


def points2uv_batch(points, cam_instrinsic_batch, calib_lidar2cam_batch, with_depth=False):
    cam_instrinsic = cam_instrinsic_batch.type(torch.float32)
    calib_lidar2cam = calib_lidar2cam_batch.type(torch.float32)
    # 先将点从雷达坐标系转换到相机坐标系
    r_velo2cam, t_velo2cam = calib_lidar2cam[:, :3, :3], calib_lidar2cam[:, :3, 3].reshape(-1, 3, 1)
    points = r_velo2cam @ points.permute(0, 2, 1) + t_velo2cam
    
    # 再将相机坐标系的3D点云转换到图像上的2D点云
    r_cam_instrinsic = cam_instrinsic[:, :3, :3]
    t_cam_instrinstic = cam_instrinsic[:, :3, 3].reshape(-1, 3, 1)
    point_2d = (r_cam_instrinsic @ points + t_cam_instrinstic).permute(0, 2, 1)
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    
    if with_depth:
        return np.concatenate([point_2d_res, point_2d[..., 2:3]], axis=-1)
    return point_2d_res


class DeformableCrossAttentionImgWithPoint(nn.Module):
    """
    
    """
    def __init__(
        self,
        dim_model=256,
        dim_single_head=64,
        dropout=0.3,
        n_levels=3,
        n_heads=6,
        n_points=9,
        out_sample_loc=False,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(
            dim_model,
            dim_single_head,
            n_levels,
            n_heads,
            n_points,
            out_sample_loc=out_sample_loc)
        self.dropout = nn.Dropout(dropout)
        self.out_sample_loc = out_sample_loc

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        input_dict,
        # query_pos=None,
        # reference_points=None,
        # src_spatial_shapes=None,
        # level_start_index=None,
        src_padding_mask=None,
        idx=None,
    ):
        tgt = input_dict["x"]
        img_feature = input_dict["img_feature"]
        pe = input_dict["pe"]
        proxy_xyz = input_dict["coor"]
        
        # ===== get uv points =========
        batch_cam_instrinsic, batch_calib_lidar2cam = input_dict["cam2img"], input_dict["lidar2cam"]
        uv_points = points2uv_batch(proxy_xyz, batch_cam_instrinsic, batch_calib_lidar2cam)

        src, spatial_shapes, center_pos = [], [], []
        for i in range(len(img_feature)):
            # feature map generation
            batch, _, cur_h, cur_w = img_feature[i].shape
            src.append(img_feature[i].reshape(batch, -1, cur_h * cur_w).transpose(2, 1).contiguous())
            spatial_shapes.append((cur_h, cur_w))
            
            # 2D query point generation
            cur_rescale = torch.tensor((cur_h, cur_w)) / torch.tensor(input_dict["img_shape"])
            cur_rescale = cur_rescale.flip(0)
            center_pos.append(uv_points * cur_rescale.cuda())
        
        src = torch.cat(src, dim=1)  # B ,sum(H*W), C
        spatial_shapes = torch.as_tensor(
            spatial_shapes,
            dtype=torch.long,
            device=tgt.device,
        )
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))
        center_pos = torch.stack(center_pos, dim=2)
        
        # cross attention
        tgt2, sampling_locations = self.cross_attn(
            self.with_pos_embed(tgt, pe),
            center_pos,
            src,
            spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        tgt = self.dropout(tgt2)

        if self.out_sample_loc:
            return tgt, sampling_locations
        else:
            return tgt
        

class DeformableTransformerDecoderAIGC(nn.Module):
    """Deformable transformer decoder.

    Note that the ``DeformableDetrTransformerDecoder`` in MMDet has different
    interfaces in multi-head-attention which is customized here. For example,
    'embed_dims' is not a position argument in our customized multi-head-self-
    attention, but is required in MMDet. Thus, we can not directly use the
    ``DeformableDetrTransformerDecoder`` in MMDET.
    """

    def __init__(
        self,
        dim,
        n_levels=3,
        depth=2,
        n_heads=4,
        dim_single_head=32,
        dim_ffn=256,
        dropout=0.0,
        out_attention=False,
        n_points=9,
    ):
        super().__init__()
        self.out_attention = out_attention
        self.layers = nn.ModuleList([])
        self.depth = depth
        self.n_levels = n_levels
        self.n_points = n_points

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        dim,
                        SelfAttention(
                            dim,
                            n_heads=n_heads,
                            dim_single_head=dim_single_head,
                            dropout=dropout,
                            out_attention=self.out_attention,
                        ),
                    ),
                    PreNorm(
                        dim,
                        DeformableCrossAttention(
                            dim,
                            dim_single_head,
                            n_levels=n_levels,
                            n_heads=n_heads,
                            dropout=dropout,
                            n_points=n_points,
                            out_sample_loc=self.out_attention,
                        ),
                    ),
                    PreNorm(dim, FFN(dim, dim_ffn, dropout=dropout)),
                ]))

    def forward(self, x, pos_embedding, src, src_spatial_shapes,
                level_start_index, center_pos):
        if self.out_attention:
            out_cross_attention_list = []
        if pos_embedding is not None:
            center_pos_embedding = pos_embedding(center_pos[..., 0, :])  # 用第一个level的2D点生成position embedding即可
        # reference_points = center_pos[:, :,
        #                               None, :].repeat(1, 1, self.n_levels, 1)
        reference_points = center_pos
        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            if self.out_attention:
                if center_pos_embedding is not None:
                    x_att, self_att = self_attn(x + center_pos_embedding)
                    x = x_att + x
                    x_att, cross_att = cross_attn(
                        x,
                        src,
                        query_pos=center_pos_embedding,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                else:
                    x_att, self_att = self_attn(x)
                    x = x_att + x
                    x_att, cross_att = cross_attn(
                        x,
                        src,
                        query_pos=None,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                out_cross_attention_list.append(cross_att)
            else:
                if center_pos_embedding is not None:
                    x_att = self_attn(x + center_pos_embedding)
                    x = x_att + x
                    x_att = cross_attn(
                        x,
                        src,
                        query_pos=center_pos_embedding,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                else:
                    x_att = self_attn(x)
                    x = x_att + x
                    x_att = cross_attn(
                        x,
                        src,
                        query_pos=None,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )

            x = x_att + x
            x = ff(x) + x

        out_dict = {'ct_feat': x}
        if self.out_attention:
            out_dict.update({
                'out_attention':
                torch.stack(out_cross_attention_list, dim=2)
            })
        return out_dict


class DeformableTransformerDecoderAIGCV20(nn.Module):
    """
    将所有点云的self-attention转成先做KNN，然后每个点只和KNN得到的点做cross-attention
    """

    def __init__(
        self,
        dim,
        n_levels=3,
        depth=2,
        n_heads=4,
        dim_single_head=32,
        dim_ffn=256,
        dropout=0.0,
        out_attention=False,
        n_points=9,
        neighbor_num_list=[16, 64],
    ):
        super().__init__()
        self.out_attention = out_attention
        self.layers = nn.ModuleList([])
        self.depth = depth
        self.n_levels = n_levels
        self.n_points = n_points
        self.neighbor_num_list = neighbor_num_list

        for i in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        dim,
                        SelfAttentionKNN(
                            dim,
                            n_heads=n_heads,
                            dim_single_head=dim_single_head,
                            dropout=dropout,
                            out_attention=self.out_attention,
                            neighbor_num=self.neighbor_num_list[i]
                        ),
                    ),
                    PreNorm(
                        dim,
                        DeformableCrossAttention(
                            dim,
                            dim_single_head,
                            n_levels=n_levels,
                            n_heads=n_heads,
                            dropout=dropout,
                            n_points=n_points,
                            out_sample_loc=self.out_attention,
                        ),
                    ),
                    PreNorm(dim, FFN(dim, dim_ffn, dropout=dropout)),
                ]))

    def forward(self, x, pos_embedding, src, src_spatial_shapes,
                level_start_index, center_pos, center_pos3d):
        if self.out_attention:
            out_cross_attention_list = []
        if pos_embedding is not None:
            center_pos_embedding = pos_embedding(center_pos[..., 0, :])  # 用第一个level的2D点生成position embedding即可
        # reference_points = center_pos[:, :,
        #                               None, :].repeat(1, 1, self.n_levels, 1)
        reference_points = center_pos
        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            if self.out_attention:
                if center_pos_embedding is not None:
                    x_att, self_att = self_attn(x + center_pos_embedding)
                    x = x_att + x
                    x_att, cross_att = cross_attn(
                        x,
                        src,
                        query_pos=center_pos_embedding,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                else:
                    x_att, self_att = self_attn(x)
                    x = x_att + x
                    x_att, cross_att = cross_attn(
                        x,
                        src,
                        query_pos=None,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                out_cross_attention_list.append(cross_att)
            else:
                if center_pos_embedding is not None:
                    # 为了防止第二维被LayerNorm, 所以第二维写一个None
                    x_att = self_attn(x + center_pos_embedding, None, center_pos3d=center_pos3d)
                    x = x_att + x
                    x_att = cross_attn(
                        x,
                        src,
                        query_pos=center_pos_embedding,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                else:
                    x_att = self_attn(x)
                    x = x_att + x
                    x_att = cross_attn(
                        x,
                        src,
                        query_pos=None,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )

            x = x_att + x
            x = ff(x) + x

        out_dict = {'ct_feat': x}
        if self.out_attention:
            out_dict.update({
                'out_attention':
                torch.stack(out_cross_attention_list, dim=2)
            })
        return out_dict

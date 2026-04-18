import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import torch.nn as nn
import torch

from torch.nn.init import trunc_normal_

from swinvar.preprocess.parameters import windows_size, CHANNEL_SIZE, VARIANT_SIZE
from swinvar.models.droppath import DropPath


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows


def window_reverse(windows, window_size, H, W):
    # (b*8*8, 7, 7, 96), 7, 56, 56
    B = int(windows.shape[0] / (H * W / window_size / window_size)) # (b*8*8 / (56 * 56 / 7 / 7)) -> b
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1) # (b, 8, 8, 7, 7, 96)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1) # (b, 8, 7, 8, 7, 96) -> (b, 56, 56, 96)

    return x


class MLP(nn.Module):
    def __init__(self, dim, inter_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def __init__(self, dim, n_activated_experts, n_expert_groups, n_limited_groups, score_func, route_scale, n_routed_experts):
        super().__init__()
        self.dim = dim
        self.topk = n_activated_experts
        self.n_groups = n_expert_groups
        self.topk_groups = n_limited_groups
        self.score_func = score_func
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(n_routed_experts, dim))
        self.bias = nn.Parameter(torch.empty(n_routed_experts))

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier 更适合 softmax/sigmoid 一类（可按激活换 he/kaiming）
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        scores = F.linear(x, self.weight)

        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()

        original_scores = scores
        scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    def __init__(self, dim, inter_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, dim, n_routed_experts, n_activated_experts, n_expert_groups, n_limited_groups, score_func, route_scale, moe_inter_dim, n_shared_experts):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts

        self.gate = Gate(self.dim, self.n_activated_experts, n_expert_groups, n_limited_groups, score_func, route_scale, self.n_routed_experts)
        self.experts = nn.ModuleList([Expert(self.dim, moe_inter_dim) for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(self.dim, n_shared_experts * moe_inter_dim)

    def forward(self, x):
        shape = x.size()
        x = x.contiguous().view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)

        # 逐个 top-k 槽位循环，而不是逐 expert
        for slot in range(self.n_activated_experts):
            expert_idx = indices[:, slot]      # (T,)
            w = weights[:, slot:slot+1]        # (T, 1)

            # 对当前 slot 上每个 expert 的 token 做分组
            for i in range(self.n_routed_experts):
                mask = (expert_idx == i)
                if not mask.any():
                    continue
                expert = self.experts[i]
                x_i = x[mask]
                y[mask] += expert(x_i) * w[mask]
        
        z = self.shared_experts(x)
        return (y + z).view(shape)


class PatchEmbed(nn.Module):
    def __init__(self, feature_size, patch_size, in_chans, embed_dim, norm_layer=nn.RMSNorm):
        super().__init__()
        self.feature_size = feature_size # (43, 18)
        self.patch_size = (patch_size, patch_size) # (2, 2)
        self.patches_resolution = [feature_size[0] // self.patch_size[0], feature_size[1] // self.patch_size[1]] # (21, 9)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1] # 21 * 9 = 189

        self.in_chans = in_chans # 3
        self.embed_dim = embed_dim # 192

        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.center_prior = CenterPrior2D(H=self.patches_resolution[0], W=self.patches_resolution[1], sigma_scale=6.0)
        self.norm = norm_layer(embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape # (b, 3, 43, 18)
        assert H == self.feature_size[0] and W == self.feature_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.feature_size[0]}*{self.feature_size[1]})."
        
        x = self.proj(x) # (b, 3, 43, 18) -> (b, 192, 21, 9)
        x = self.center_prior(x)
        x = x.flatten(2).transpose(1, 2) # (b, 192, 21, 9) -> (b, 192, 21*9) -> (b, 21*9, 192)
        x = self.norm(x) # (b, 21*9, 192)

        return x # (b, 21*9, 192)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)) # (13*13, num_heads)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij")) # (2, 7, 7)
        coords_flatten = torch.flatten(coords, 1) # (2, 49)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # (2, 49, 49)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # (49, 49, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1) # (49, 49)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None): # (b*8*8, 49, 96)
        B_, N, C = x.shape # (b*8*8, 49, 96)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # (b*8*8, 49, 96) -> (b*8*8, 49, 96*3) -> (b*8*8, 49, 3, num_heads, 96//num_heads) -> (3, b*8*8, num_heads, 49, 96//num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2] # (b*8*8, num_heads, 49, 96//num_heads)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # (b*8*8, num_heads, 49, 96//num_heads) @ (b*8*8, num_heads, 96//num_heads, 49) -> (b*8*8, num_heads, 49, 49)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        ) # (49*49, num_heads) -> (49, 49, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # (num_heads, 49, 49)
        attn = attn + relative_position_bias.unsqueeze(0) # (b*8*8, num_heads, 49, 49)

        if mask is not None:
            nW = mask.shape[0] # (1*8*8, 49, 49)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # (b, 8*8, num_heads, 49, 49)
            attn = attn.view(-1, self.num_heads, N, N) # (b*8*8, num_heads, 49, 49)
            attn = self.softmax(attn) # (b*8*8, num_heads, 49, 49)
        else:
            attn = self.softmax(attn) # (b*8*8, num_heads, 49, 49)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # (b*8*8, num_heads, 49, 49) @ (b*8*8, num_heads, 49, 96//num_heads) -> (b*8*8, num_heads, 49, 96//num_heads) -> (b*8*8, 49, num_heads, 96//num_heads)
        # (b*8*8, 49, 96)
        x = self.proj(x) # (b*8*8, 49, 96)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class CenterPrior2D(nn.Module):
    """
    对特征图的“位置轴”加中心区域 bias。
    适用于输入形状 (B, C, H, W)，比如 (B, 192, 21, 9)。

    思路：
    - 先在 H 维上构造一个 1D 高斯型中心权重 prior_1d[h] ∈ [0,1]，
      在中心行最大，越靠边越小；
    - 再 broadcast 到 (H, W) 得到 center_prior_2d[h, w]；
    - forward 时用一个可学习系数 gamma 做缩放：
        x_scaled = x * (1 + gamma * center_prior_2d)
      gamma 初始为 0，不会破坏你原来的模型，
      训练过程中如果中心先验有用，gamma 会自动学成正数。
    """
    def __init__(self, H: int, W: int, sigma_scale: float = 6.0):
        """
        参数：
        - H: 特征图高度（位置轴长度），比如 21
        - W: 特征图宽度（事件特征轴长度），比如 9
        - sigma_scale: 控制中心区域宽度，越小越“尖锐”
        """
        super().__init__()
        self.H = H
        self.W = W

        # ---- 构造只沿 H 变化的 1D 高斯先验 ----
        center_h = H // 2               # 位置轴的中心行
        yy = torch.arange(H).float()    # (H,)

        # sigma 控制中心区域宽度，可以先用 H/6 这个经验值
        sigma_h = H / sigma_scale

        # 高斯距离
        dist2 = (yy - center_h) ** 2 / (sigma_h ** 2)   # (H,)
        prior_1d = torch.exp(-0.5 * dist2)              # (H,)
        prior_1d = prior_1d / prior_1d.max()            # 归一到 [0,1]

        # broadcast 到 (H, W)：同一行的 W 个位置权重相同
        center_prior_2d = prior_1d.view(H, 1).expand(H, W)  # (H, W)

        # 作为 buffer 注册，不参与梯度
        self.register_buffer("center_prior_2d", center_prior_2d)  # (H, W)

        # 可学习缩放系数 gamma，初始为 0：一开始不影响模型
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        返回：加了中心 bias 的特征 (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert H == self.H and W == self.W, \
            f"CenterPrior2D 期望输入尺寸为 (B, C, {self.H}, {self.W})，但得到的是 (B, C, {H}, {W})"

        # (H, W) -> (1, 1, H, W)，然后 broadcast 到 batch 和通道
        cp = self.center_prior_2d.to(x.device)           # (H, W)
        cp = cp.unsqueeze(0).unsqueeze(0)                # (1, 1, H, W)
        # broadcast：中心位置所有通道同样缩放
        # scale = 1 + gamma * cp，初始 gamma=0 => scale=1
        scale = 1.0 + self.gamma * cp                    # (1, 1, H, W)
        # 自动 broadcast 到 (B, C, H, W)
        x = x * scale

        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size,
        n_routed_experts,
        n_activated_experts,
        n_expert_groups,
        n_limited_groups,
        score_func,
        route_scale,
        moe_inter_dim,
        n_shared_experts,
        shift_size=0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.RMSNorm,
        fused_window_process=False,
    ):
        super().__init__()
        self.dim = dim # 192
        self.input_resolution = input_resolution # (21, 9)
        self.num_heads = num_heads # 3
        self.window_size = window_size # 3
        self.shift_size = shift_size # 0
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim) # 96
        self.moe = MoE(dim=dim, n_routed_experts=n_routed_experts, n_activated_experts=n_activated_experts, n_expert_groups=n_expert_groups, n_limited_groups=n_limited_groups, score_func=score_func, route_scale=route_scale, moe_inter_dim=moe_inter_dim, n_shared_experts=n_shared_experts)

        if self.shift_size > 0:
            # SW-MSA
            H, W = self.input_resolution
            feature_mask = torch.zeros((1, H, W, 1)) # (1, 56, 56, 1)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    feature_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(feature_mask, self.window_size) # (1, 8, 8, 7, 7, 1) -> (1*8*8, 7, 7, 1)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # (1*8*8, 49)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # (1*8*8, 1, 49) - (1*8*8, 49, 1) -> (1*8*8, 49, 49)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x): # (b, 21*9, 192)
        H, W = self.input_resolution # (56, 56)
        B, L, C = x.shape # (b, 56*56, 96)
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C) # (b, 56, 56, 96)

        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                x_windows = window_partition(shifted_x, self.window_size) # (b*8*8, 7, 7, 96)
            # else:
            #     x_windows = WindowProcee.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) # (b*8*8, 49, 96)

        attn_windows = self.attn(x_windows, mask=self.attn_mask) # (b*8*8, 49, 96)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) # (b*8*8, 7, 7, 96)

        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W) # (b, 56, 56, 96)
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) # (b, 56, 56, 96)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W) # (b, 56, 56, 96)
            x = shifted_x # (b, 56, 56, 96)
        x = x.view(B, H * W, C) # (b, 56*56, 96)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.moe(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}"


class BasicLayer(nn.Module):

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        n_routed_experts,
        n_activated_experts,
        n_expert_groups,
        n_limited_groups,
        score_func,
        route_scale,
        moe_inter_dim,
        n_shared_experts,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.RMSNorm,
        qk_scale=None,
        use_checkpoint=False,
        qkv_bias=True,
        fused_window_process=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, 
                                 n_routed_experts=n_routed_experts, n_activated_experts=n_activated_experts,
                                 n_expert_groups=n_expert_groups, n_limited_groups=n_limited_groups, score_func=score_func,
                                 route_scale=route_scale, moe_inter_dim=moe_inter_dim, n_shared_experts=n_shared_experts,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, fused_window_process=fused_window_process)
                                 for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class SwinVar(nn.Module):

    def __init__(
        self,
        num_classes,
        embed_dim=96,
        patch_size=4,
        in_chans=3,
        window_size=7,
        n_routed_experts=10,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=2,
        score_func="sigmoid",
        route_scale=1,
        moe_inter_dim=64,
        n_shared_experts=1,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.,
        feature_size=(windows_size, CHANNEL_SIZE),
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        norm_layer=nn.RMSNorm,
        patch_norm=True,
        ape=False,
        qkv_bias=True,
        qk_scale=None,
        use_checkpoint=False,
        fused_window_process=False
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        # self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features = embed_dim

        self.patch_embed = PatchEmbed(feature_size=feature_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        num_patches = self.patch_embed.num_patches # 21 * 9 = 189
        self.patches_resolution = self.patch_embed.patches_resolution # (21, 9)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 1 ** i_layer),
                input_resolution=(self.patches_resolution[0] // (1 ** i_layer), self.patches_resolution[1] // (1 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                n_routed_experts=n_routed_experts,
                n_activated_experts=n_activated_experts,
                n_expert_groups=n_expert_groups,
                n_limited_groups=n_limited_groups,
                score_func=score_func,
                route_scale=route_scale,
                moe_inter_dim=moe_inter_dim,
                n_shared_experts=n_shared_experts,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[: i_layer]): sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                fused_window_process=fused_window_process
            )
            self.layers.append(layer)
        
        self.norm2 = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head1 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head2 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head3 = nn.Linear(self.num_features, 4) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.RMSNorm):
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def forward_features(self, x):
        # x: (b, 3, 43, 18)
        x = self.patch_embed(x) # (b, 21*9, 192)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        
        x = self.norm2(x) # (b, 56/2 * 56/2, 96*2)
        x = self.avgpool(x.transpose(1, 2)) # (b, 96*2, 56/2 * 56/2) -> # (b, 96*2, 1)
        x = torch.flatten(x, 1) # (b, 96*2)

        return x
    
    def forward(self, x=None, **kwargs):
        # x: (b, 43, 3, 18)
        if x is None and "input_ids" in kwargs:
            x = kwargs["input_ids"]
        x = x.permute(0, 2, 1, 3).contiguous() # x: (b, 3, 43, 18)
        x = self.forward_features(x)
        x_1 = self.head1(x)
        x_2 = self.head2(x)
        x_3 = self.head3(x)

        return x_1, x_2, x_3


if __name__ == "__main__":

    train_input_path = ["/data2/lijie/result/Transformer_pileup_3_channel/HG003_WES", "/data2/lijie/result/Transformer_pileup_3_channel/HG004_WES", "/data2/lijie/result/Transformer_pileup_3_channel/HG005_WES"]
    output_path = "/data2/lijie/result/Transformer_pileup_3_channel/HG003_4_5to2_WES"
    data_path = "/data2/lijie/data/GRCh38"
    test_input_path = "/data2/lijie/result/Transformer_pileup_3_channel/HG002_WES"

    args = {
        "input_path": train_input_path,
        "output_path": output_path,
        "reference_path": "/data2/lijie/reference/GRCh38_full_plus_hs38d1_analysis_set_minus_alts/GRCh38_full_plus_hs38d1_analysis_set_minus_alts.fa",
        "bam_path_train": f"{data_path}/HG005/HG005.bam",
        "bed_path_train": f"{data_path}/HG005/HG005.bed",
        "vcf_path_train": f"{data_path}/HG005/HG005.vcf",
        "bam_path_test": f"{data_path}/HG003/HG003.bam",
        "bed_path_test": f"{data_path}/HG003/agilent_sureselect_human_all_exon_v5_b37_targets.bed",
        "vcf_path_test": f"{data_path}/HG003/HG003.vcf",
        "file": "balance",
        "FT_file": "pileup",
        "Fine-tuning": False,
        "train_ratio": 0.8,
        "seed": 22,
        "epochs": 300,
        "batch_size": 600,
        "feature_size": (windows_size, CHANNEL_SIZE),
        "num_classes": VARIANT_SIZE,
        "depths": [2, 6, 2],
        "num_heads": [6, 6, 6],
        "embed_dim": 192,
        "patch_size": 2,
        "window_size": 3,
        "n_routed_experts": 8,
        "n_activated_experts": 2,
        "n_expert_groups": 1,
        "n_limited_groups": 1,
        "score_func": "sigmoid",
        "route_scale": 1,
        "moe_inter_dim": 48,
        "n_shared_experts": 1,
        "drop_rate": 0.1,
        "drop_path_rate": 0.1,
        "attn_drop_rate": 0.1,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "pct_start": 0.3,
        "factor": [1, 1],
        "num_workers": 60,
        "patience": 100,
        "model_save_path": "best_model.pth",
        "log_file_train": "train_log.txt",
        "log_file_test": "test_log.txt",
        "matplot_save_path": "train_process.png",
        "hyperparams_log": "hyperparams.xlsx",
        "test_batch_size": 5000,
        "test_input_path": test_input_path,
    }


    # x = torch.rand(1, 3, windows_size, CHANNEL_SIZE)
    model = SwinVar(
        feature_size=args["feature_size"],
        num_classes=args["num_classes"],
        embed_dim=args["embed_dim"],
        patch_size=args["patch_size"],
        window_size=args["window_size"],
        n_routed_experts=args["n_routed_experts"],
        n_activated_experts=args["n_activated_experts"],
        n_expert_groups=args["n_expert_groups"],
        n_limited_groups=args["n_limited_groups"],
        score_func=args["score_func"],
        route_scale=args["route_scale"],
        moe_inter_dim=args["moe_inter_dim"],
        n_shared_experts=args["n_shared_experts"],
        depths=args["depths"],
        num_heads=args["num_heads"],
        drop_rate=args["drop_rate"],
        drop_path_rate=args["drop_path_rate"],
        attn_drop_rate=args["attn_drop_rate"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型的总参数数目: {total_params}")
    # print(model)
    
    # from lora import apply_lora_to_model, finetunning

    # replace_list = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]

    # for name, module in replace_list:
    #     parts = name.split(".")
    #     parent = model
    #     for p in parts[:-1]:
    #         parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    #     setattr(parent, parts[-1], LoRA(module, 4, 4))


    # for name, param in model.named_parameters():
    #     if "lora_" not in name:
    #         param.requires_grad = False

    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"模型的总参数数目: {total_params}")

    # model.load_state_dict(torch.load("/data2/lijie/result/Transformer_pileup_3_channel/HG003_4_5to2_WES/train_moe/balance/best_model.pth"))

    # # apply_lora_to_model(model)
    # finetunning(model)

    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"模型的总参数数目: {total_params}")
    # print(model)










    # output_path = "/data/lijie/postgraduate/result/Transformer_pileup"
    # data_path = "/data2/lijie/data/GRCh38"

    # args = {
    #     "input_path": f"{output_path}",
    #     "reference_path": "/data2/lijie/reference/GRCh38_full_plus_hs38d1_analysis_set_minus_alts/GRCh38_full_plus_hs38d1_analysis_set_minus_alts.fa",
    #     "bam_path_train": f"{data_path}/HG005/HG005.bam",
    #     "bed_path_train": f"{data_path}/HG005/HG005.bed",
    #     "vcf_path_train": f"{data_path}/HG005/HG005.vcf",
    #     "bam_path_test": f"{data_path}/HG003/HG003.bam",
    #     "bed_path_test": f"{data_path}/HG003/agilent_sureselect_human_all_exon_v5_b37_targets.bed",
    #     "vcf_path_test": f"{data_path}/HG003/HG003.vcf",
    #     "max_len": 100,
    #     "vocab_size": 7,
    #     "train_ratio": 0.8,
    #     "seed": 22,
    #     "epochs": 1000,
    #     "batch_size": 3000,
    #     "accumulation": 4,
    #     "input_dim": CHANNEL_SIZE,
    #     "d_model": 256,
    #     "n_head": 4,
    #     "dim_feedforward": 512,
    #     "num_layers": 4,
    #     "dropout": [0.2, 0.2],
    #     "lr": 0.0001,
    #     "weight_decay": 0.01,
    #     "pct_start": 0.3,
    #     "factor": [25, 1000],
    #     "num_workers": 60,
    #     "patience": 90,
    #     "model_save_path": f"{output_path}/best_model.pth",
    #     "log_file_train": f"{output_path}/train_log.txt",
    #     "log_file_test": f"{output_path}/test_log_.txt",
    #     "matplot_save_path": f"{output_path}/train_process.png"
    # }

    # model = SwinTransformer(
    #     args["input_dim"], args["d_model"], args["n_head"], args["dim_feedforward"], args["num_layers"], args["max_len"]
    # )
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"模型的总参数数目: {total_params}")

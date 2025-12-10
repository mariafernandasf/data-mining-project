"""
This code was originally obtained from:
https://github.com/facebookresearch/deit
and
https://github.com/meta-llama/codellama/blob/main/llama/model.py

IEOR4540: I took this from https://github.com/naver-ai/rope-vit/tree/main/deit

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import cg

from functools import partial
from typing import Tuple

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from models_v2 import vit_models, Layer_scale_init_Block, Attention
from models_v2_rope import rope_vit_models, apply_rotary_emb

def get_I_sparse(head_dim: int) -> torch.sparse.Tensor:
    indices = torch.arange(head_dim, dtype=torch.int64).unsqueeze(0).repeat(2, 1)
    values = torch.ones(head_dim)
    I_sparse = torch.sparse_coo_tensor(indices, values, size=(head_dim, head_dim))
    return I_sparse



class CayleySTRINGAttention(Attention):
    """Multi-head Attention block with Cayley-STRING embeddings."""
    def __init__ (self, dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop,
                  reflection_variant=False, sparse_variant_fixed_f=False, sparsity_f=0.8):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
                
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.reflection_variant = reflection_variant
        self.sparse_variant_fixed_f = sparse_variant_fixed_f
        self.sparsity_f = sparsity_f

        if self.reflection_variant:
            n = 0.01 * torch.randn(num_heads, head_dim)
            self.n_reflect = nn.Parameter(n)
        elif self.sparse_variant_fixed_f:
            n_total = head_dim * (head_dim -1) //2 # upper triangular size
            n_keep = int(self.sparsity_f * head_dim * head_dim) # total nonzero entries
            
            # when we create a skew-symmetric matrix, the number of non-zeros doubles
            n_keep = n_keep //2  
            n_keep = min(n_keep, n_total) 

            # upper triangular indices
            upper_indices = torch.triu_indices(head_dim, head_dim, offset=1)
            mask = torch.randperm(n_total)[:n_keep] # random indices to keep
            row = upper_indices[0][mask]
            col = upper_indices[1][mask]

            self.register_buffer("row", row)
            self.register_buffer("col", col)

            nnz = row.numel()
            self.nnz_values = nn.Parameter(torch.randn(num_heads, nnz))
        else: 
            # S = M - M^T to ensure S is antisymmetric
            M = 0.01 * torch.randn(num_heads, head_dim, head_dim)
            self.M_cayley = nn.Parameter(M)
        
    def S_cayley(self):
        """Antisymmetric matrix: S = M - M^T
        This ensures that S is remains antisymmetric after 
        each update to M_cayley
        """
        return self.M_cayley - self.M_cayley.transpose(-1, -2)
    
    def S_cayley_sparse_fixed_f(self, head):        
        nnz_values_head = self.nnz_values[head]  # (n_nonzero,)

        # Create COO tensor directly
        indices = torch.stack([self.row, self.col], dim=0)  # (2, nnz)
        M_head = torch.sparse_coo_tensor(
            indices,
            nnz_values_head,
            size=(self.head_dim, self.head_dim),
            device=nnz_values_head.device,
            dtype=nnz_values_head.dtype
        )
        # enforce antisymmetry: S = M - M^T (COO)
        S_head = M_head - M_head.transpose(0, 1)
        return S_head.coalesce()


    def apply_sparse_cayley_fixed_f(self, z):
        B, num_heads, N, head_dim = z.shape
        out = torch.empty_like(z)
        I = get_I_sparse(head_dim).to(device = z.device, dtype = torch.float32) # (head_dim, head_dim)
        original_dtype = z.dtype
        for head in range(num_heads):
            S_head = self.S_cayley_sparse_fixed_f(head).to(device = z.device, dtype = torch.float32)
            z_head = z[:, head, :,:]
            z_head = z_head.reshape(B*N, head_dim)
            
            I_minus_S = I - S_head
            I_plus_S = I + S_head

            rhs = z_head.t().to(torch.float32)  # (head_dim, B*N)
            
            original_dtype = z_head.dtype
            A = I_plus_S.to(torch.float32)                 # (head_dim, head_dim)
            x_head = torch.linalg.solve(A, rhs)           # (head_dim, B*N), x_head is dense

            x_head = torch.clamp(x_head, -65500, 65500).to(original_dtype)
            result = torch.sparse.mm(I_minus_S, x_head) # I_minus_S is sparse
            result = result.t().reshape(B, N, head_dim)  
            
            out[:, head,:, :] = result

        return out
    
    def apply_cayley_reflection_variant(self, z, eps = 1e-8):
        B, num_heads, N, head_dim = z.shape
        z_reshaped = z.permute(0,2,1,3).reshape(B*N, num_heads, head_dim)
        I = torch.eye(head_dim, device=z.device, dtype=z.dtype) # (head_dim, head_dim)

        n = self.n_reflect.to(z.dtype)  # (num_heads, head_dim)
        # n n^T: (num_heads, head_dim, 1) @ (num_heads, 1, head_dim) = (num_heads, head_dim, head_dim)
        n_outer = torch.einsum('hd,he->hde', n, n)  # (num_heads, head_dim, head_dim)
        
        # compute squared L2 norm of n for each head
        n_norm_sq = torch.sum(n * n, dim=1, keepdim=True)  # (num_heads, 1)
        
        # v = v_ort + v_par
        # v_par = v - v_ort
        # v_ort = proj_n(z) = (n^T z) n / ||n||^2
        # SO, the projected vector Rv = v_par - v_ort
        # = v - v_ort - v_ort = v - 2 v_ort
        # = (I - 2 v_ort) v
        # = (I - 2 (n n^T) / ||n||^2 ) v
        # = P_reflect v
        P_reflect = I.unsqueeze(0) - 2 * n_outer / (n_norm_sq.unsqueeze(-1) + eps)
    
        z_reshaped_heads = z_reshaped.permute(1, 2, 0)  # (B*N, H, D) -> (H, D, B*N)
        result = torch.einsum('hde,heb->hdb', P_reflect, z_reshaped_heads)  # (H, D, B*N)
        result = result.permute(2, 0, 1).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        return result
    

    def apply_regular_cayley(self, z):
        B, num_heads, N, head_dim = z.shape
        z_reshaped = z.permute(0,2,1,3).reshape(B*N, num_heads, head_dim)
        I = torch.eye(head_dim, device=z.device, dtype=z.dtype) # (head_dim, head_dim)

        S = self.S_cayley().to(z.dtype) # (num_heads, head_dim, head_dim)
            
        # (I + S)
        # S = (num_heads, head_dim, head_dim), I = (head_dim, head_dim)
        # need to first unsqueeze I so that I = (1, head_dim, head_dim)
        # and dimensions match
        I_plus_S = I.unsqueeze(0) + S

        # (I - S)
        # need to first unsqueeze I so that dimensions match
        I_minus_S = I.unsqueeze(0) - S

        # x = (I + S)^{-1} z, use linear solver head-wise to avoid huge expansion
        # torch.linalg.solve doesn't support float16, so solve in float32
        original_dtype = z_reshaped.dtype
        A = I_plus_S.to(torch.float32)                 # (num_heads, head_dim, head_dim)
        rhs = z_reshaped.permute(1, 2, 0).to(torch.float32)  # (num_heads, head_dim, B*N)
        x_heads = torch.linalg.solve(A, rhs)           # (num_heads, head_dim, B*N)
        x = x_heads.permute(2, 0, 1)                   # (B*N, num_heads, head_dim)
        x = torch.clamp(x, -65500, 65500).to(original_dtype)

        # result = (I - S) @ x = P_cayley @ x
        result = torch.einsum('hde,bhe->bhd', I_minus_S, x)
        result = result.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        return result
    

    def apply_cayley(self, z, eps = 1e-8):
        if self.reflection_variant:
            return self.apply_cayley_reflection_variant(z, eps=eps)
        
        elif self.sparse_variant_fixed_f:
            return self.apply_sparse_cayley_fixed_f(z)
        
        return self.apply_regular_cayley(z)


    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        
        q[:, :, 1:] = self.apply_cayley(q[:, :, 1:])
        k[:, :, 1:] = self.apply_cayley(k[:, :, 1:])

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
class Cayley_STRING_Layer_scale_init_Block(Layer_scale_init_Block):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, freqs_cis):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), freqs_cis=freqs_cis))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        
        return x

class cayley_STRING_vit_models(rope_vit_models):
    def __init__(self, rope_theta,
                 **kwargs):
        super().__init__(rope_theta = rope_theta, rope_mixed = True, use_ape = False, **kwargs)
                


# -------- --- 16x16 Patch Models ------------
# Regular Cayley-STRING 16x16
@register_model
def cayleySTRING_regular_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, Attention_block=CayleySTRINGAttention,
        rope_theta=100.0, **kwargs)
    model.default_cfg = _cfg()
    return model

# Cayley-STRING reflection variant 16x16
@register_model
def cayleySTRING_reflection_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, 
        Attention_block=partial(CayleySTRINGAttention, 
                                reflection_variant=True),
        rope_theta=100.0, **kwargs)
    model.default_cfg = _cfg()
    return model

# Cayley-STRING sparse-variant fixed f=20%
@register_model
def cayleySTRING_sparse_fixed20pct_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, 
        Attention_block=partial(CayleySTRINGAttention, 
                                sparse_variant_fixed_f=True,
                                sparsity_f = 0.2),
        rope_theta=100.0, **kwargs)
    model.default_cfg = _cfg()
    return model

# ------------ 8x8 Patch Models ------------
# Regular Cayley-STRING 8x8
@register_model
def cayleySTRING_regular_deit_small_patch8_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, Attention_block=CayleySTRINGAttention,
        rope_theta=100.0, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cayleySTRING_reflection_deit_small_patch8_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, 
        Attention_block=partial(CayleySTRINGAttention, 
                                reflection_variant=True),
        rope_theta=100.0, **kwargs)
    model.default_cfg = _cfg()
    return model

# Cayley-STRING sparse-variant fixed f=20%
@register_model
def cayleySTRING_sparse_fixed20pct_deit_small_patch8_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, 
        Attention_block=partial(CayleySTRINGAttention, 
                                sparse_variant_fixed_f=True,
                                sparsity_f = 0.2),
        rope_theta=100.0, **kwargs)
    model.default_cfg = _cfg()
    return model
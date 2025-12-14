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
                  reflection_variant=False, sparse_variant_fixed_f=False, sparsity_f=0.8,
                  use_sparse_linear_solver=False, 
                  sparse_variant_learnable=False, 
                  tau = 1.0):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
                
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.reflection_variant = reflection_variant
        self.sparse_variant_fixed_f = sparse_variant_fixed_f
        self.sparsity_f = sparsity_f
        self.use_sparse_linear_solver = use_sparse_linear_solver
        self.sparse_variant_learnable = sparse_variant_learnable

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
            # TODO: might need to change this to 0.01 * torch.randn, might cause training
            # to blow up early on
            self.nnz_values = nn.Parameter(torch.randn(num_heads, nnz))
        
        elif self.sparse_variant_learnable:
            self.values = nn.Parameter(0.01 * torch.randn(num_heads, head_dim, head_dim))
            
            # each head will have its own sparsity pattern, want expected ON probability
            # to be 50% for each cell at initialization
            self.log_alpha = nn.Parameter(-0.1 * torch.ones(num_heads, head_dim, head_dim))
            self.tau = tau 
        else: 
            # S = M - M^T to ensure S is antisymmetric
            M = 0.01 * torch.randn(num_heads, head_dim, head_dim)
            self.M_cayley = nn.Parameter(M)
    
    def set_tau(self, tau):
        self.tau = tau

    def l0_penalty(self):
        """
        Equation 12 in Louizos paper:
        Sum(probability of gate being nonzero)
        """
        mask = torch.ones_like(self.log_alpha).triu(1)
        prob = torch.sigmoid(self.log_alpha) * mask
        return prob.sum()

    def expected_nnz_pct(self):
        # expected nnz% of full S (including both triangles), off-diagonal only
        mask = torch.ones_like(self.log_alpha).triu(1)
        p = torch.sigmoid(self.log_alpha) * mask
        expected_nnz = 2.0 * p.sum(dim=(-1, -2))             # per head
        pct = expected_nnz / float(self.head_dim * self.head_dim)
        return pct.mean()  # scalar tensor

    def sparse_linear_solver(self, I_plus_S, z_head_T, B, N):
        I_plus_S = I_plus_S.to_sparse_csr()
        xs = []
        for i in range(B*N):
            rhs = z_head_T[:, i]
            x_i = torch.sparse.spsolve(I_plus_S, rhs)
            xs.append(x_i.unsqueeze(1))
        x_head = torch.cat(xs, dim=1)  # (head_dim, B*N)
        return x_head
    
    def dense_linear_solver(self, I_plus_S, z_head_T):
        A = I_plus_S.to_dense().to(torch.float32)                 # (head_dim, head_dim)
        x_head = torch.linalg.solve(A, z_head_T)           # (head_dim, B*N), x_head is dense
        return x_head    
    
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
            dtype=torch.float32
        )
        # enforce antisymmetry: S = M - M^T (COO)
        S_head = M_head - M_head.transpose(0, 1)
        return S_head.coalesce()

    def S_cayley_sparse_learnable(self):
        # upper triangular mask
        mask = torch.ones_like(self.log_alpha).triu(1)

        if self.training: 

            # uniform distribution on [0,1]
            u = torch.rand_like(self.log_alpha).clamp_(1e-6, 1 - 1e-6)
            logistic_noise = torch.log(u) - torch.log1p(-u)

            #  ~ Gumbel(log_alpha, tau)
            logits = (self.log_alpha + logistic_noise) / self.tau

            z_soft = torch.sigmoid(logits)
            z_hard = (z_soft >= 0.5).to(z_soft.dtype)
            z = z_soft + (z_hard - z_soft).detach()  
        
        else: # at eval use log_alpha, our learned mask, as the sparsity mask
            z_hard = (torch.sigmoid(self.log_alpha) > 0.5).float()
            z = z_hard

        z_ut = z * mask  # zero out lower triangular and diagonal
        z_hard_ut = z_hard * mask

        # get values for upper triangular
        S_upper = z_ut * self.values.triu(1)

        # skew-symmetrize
        S = S_upper - S_upper.transpose(-1, -2)

        with torch.no_grad():
            mask_active_ut = z_hard_ut.sum(dim=(-1, -2))
            mask_pct = ((mask_active_ut * 2.0) / (self.head_dim * self.head_dim)).mean().item()

            eps = 1e-8
            S_pct = ((S.abs() > eps).sum(dim=(-1, -2)) / (self.head_dim * self.head_dim)).mean().item()

            if self.training:
                self._last_train_mask_nnz_pct = mask_pct
                self._last_train_S_nnz_pct = S_pct
            else:
                self._last_eval_mask_nnz_pct = mask_pct
                self._last_eval_S_nnz_pct = S_pct

        return S

    def last_mask_nnz_pct(self):
        return getattr(self, "_last_train_mask_nnz_pct", None), getattr(self, "_last_eval_mask_nnz_pct", None),
    
    def last_S_nnz_pct(self): 
        return getattr(self, "_last_train_S_nnz_pct", None), getattr(self, "_last_eval_S_nnz_pct", None)
    
    def apply_sparse_cayley_fixed_f(self, z):
        B, num_heads, N, head_dim = z.shape
        original_dtype = z.dtype
        
        with torch.amp.autocast(device_type=z.device.type, enabled=False):
            # Convert once at the start
            z_f32 = z.to(torch.float32)
            I = get_I_sparse(head_dim).to(device=z.device, dtype=torch.float32)
            
            # Pre-allocate output in float32, convert at end
            out_f32 = torch.empty(B, num_heads, N, head_dim, device=z.device, dtype=torch.float32)
            
            for head in range(num_heads):
                S_head = self.S_cayley_sparse_fixed_f(head)
                z_head_T = z_f32[:, head, :, :].reshape(B*N, head_dim).t()  # (head_dim, B*N)
                
                I_plus_S = I + S_head
                I_minus_S = I - S_head
                
                if self.use_sparse_linear_solver:
                    x_head = self.sparse_linear_solver(I_plus_S, z_head_T, B, N)
                else:
                    x_head = self.dense_linear_solver(I_plus_S, z_head_T)
                
                # Direct sparse-dense mm, result is dense
                result = torch.sparse.mm(I_minus_S, x_head).t().reshape(B, N, head_dim)
                out_f32[:, head, :, :] = result
            
            # Single clamp and convert at the end
            out = torch.clamp(out_f32, -65500, 65500).to(original_dtype)

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

        if self.sparse_variant_learnable:
            S = self.S_cayley_sparse_learnable().to(z.dtype) # (num_heads, head_dim, head_dim)
        else:
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
        
        # both regular and sparse learnable variant use same function
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

# -------- cayley-STRING Sparse Variants 16x16 Patch Models ------------

# Cayley-STRING sparse-variant fixed f=45%
@register_model
def cayleySTRING_sparse_fixed45pct_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, 
        Attention_block=partial(CayleySTRINGAttention, 
                                sparse_variant_fixed_f=True,
                                sparsity_f = 0.45),
        rope_theta=100.0, **kwargs)
    model.default_cfg = _cfg()
    return model

# Cayley-STRING sparse-variant fixed f=40%
@register_model
def cayleySTRING_sparse_fixed40pct_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, 
        Attention_block=partial(CayleySTRINGAttention, 
                                sparse_variant_fixed_f=True,
                                sparsity_f = 0.4),
        rope_theta=100.0, **kwargs)
    model.default_cfg = _cfg()
    return model

# Cayley-STRING sparse-variant fixed f=35%
@register_model
def cayleySTRING_sparse_fixed35pct_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, 
        Attention_block=partial(CayleySTRINGAttention, 
                                sparse_variant_fixed_f=True,
                                sparsity_f = 0.35),
        rope_theta=100.0, **kwargs)
    model.default_cfg = _cfg()
    return model

# Cayley-STRING sparse-variant fixed f=30%
@register_model
def cayleySTRING_sparse_fixed30pct_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, 
        Attention_block=partial(CayleySTRINGAttention, 
                                sparse_variant_fixed_f=True,
                                sparsity_f = 0.3),
        rope_theta=100.0, **kwargs)
    model.default_cfg = _cfg()
    return model

# Cayley-STRING sparse-variant fixed f=25%
@register_model
def cayleySTRING_sparse_fixed25pct_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, 
        Attention_block=partial(CayleySTRINGAttention, 
                                sparse_variant_fixed_f=True,
                                sparsity_f = 0.25,
                                use_sparse_linear_solver=False),
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

# Cayley-STRING sparse-variant fixed f=15%
@register_model
def cayleySTRING_sparse_fixed15pct_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, 
        Attention_block=partial(CayleySTRINGAttention, 
                                sparse_variant_fixed_f=True,
                                sparsity_f = 0.15,
                                use_sparse_linear_solver=False),
        rope_theta=100.0, **kwargs)
    model.default_cfg = _cfg()
    return model

# Cayley-STRING sparse-variant fixed f=10%
@register_model
def cayleySTRING_sparse_fixed10pct_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, 
        Attention_block=partial(CayleySTRINGAttention, 
                                sparse_variant_fixed_f=True,
                                sparsity_f = 0.1,
                                use_sparse_linear_solver=False),
        rope_theta=100.0, **kwargs)
    model.default_cfg = _cfg()
    return model



# Cayley-STRING sparse-variant fixed f=5%
@register_model
def cayleySTRING_sparse_fixed5pct_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, 
        Attention_block=partial(CayleySTRINGAttention, 
                                sparse_variant_fixed_f=True,
                                sparsity_f = 0.05),
        rope_theta=100.0, **kwargs)
    model.default_cfg = _cfg()
    return model

# ----------------- CAYLEY STRING LEARNABLE VARIANTS -----------------
# Cayley-STRING LEARNABLE variant
@register_model
def cayleySTRING_sparse_learnable_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, 
        Attention_block=partial(CayleySTRINGAttention, 
                                sparse_variant_fixed_f=False,
                                sparse_variant_learnable=True,
                                tau = 1.0, # initialize gumbel sigmoid tau
                                use_sparse_linear_solver=False),
        rope_theta=100.0 , #sparse_variant_learnable = True, 
        **kwargs)
    model.default_cfg = _cfg()
    return model

# ------------------------ 8x8 Patch Models ------------------------
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

# Cayley-STRING sparse-variant fixed f=40%
@register_model
def cayleySTRING_sparse_fixed40pct_deit_small_patch8_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, 
        Attention_block=partial(CayleySTRINGAttention, 
                                sparse_variant_fixed_f=True,
                                sparsity_f = 0.4,
                                use_sparse_linear_solver=False),
        rope_theta=100.0, **kwargs)
    model.default_cfg = _cfg()
    return model

# Cayley-STRING sparse-variant fixed f=30%
@register_model
def cayleySTRING_sparse_fixed30pct_deit_small_patch8_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, 
        Attention_block=partial(CayleySTRINGAttention, 
                                sparse_variant_fixed_f=True,
                                sparsity_f = 0.3,
                                use_sparse_linear_solver=False),
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
                                sparsity_f = 0.2,
                                use_sparse_linear_solver=False),
        rope_theta=100.0, **kwargs)
    model.default_cfg = _cfg()
    return model

# Cayley-STRING sparse-variant fixed f=10%
@register_model
def cayleySTRING_sparse_fixed10pct_deit_small_patch8_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cayley_STRING_vit_models(
        img_size = img_size, patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Cayley_STRING_Layer_scale_init_Block, 
        Attention_block=partial(CayleySTRINGAttention, 
                                sparse_variant_fixed_f=True,
                                sparsity_f = 0.1,
                                use_sparse_linear_solver=False),
        rope_theta=100.0, **kwargs)
    model.default_cfg = _cfg()
    return model
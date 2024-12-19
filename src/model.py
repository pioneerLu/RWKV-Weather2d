import torch
import torch.nn.functional as F
import math
import numpy as np
import os, math, gc, importlib
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
from transformers import CLIPVisionModel
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop
# if os.environ["RWKV_JIT_ON"] == "1":

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method


from torch.utils.cpp_extension import load

HEAD_SIZE = 64
wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(10)}"])


    
class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)

########################################################################################################

class RWKV_Tmix_x060(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = 64
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            # print(args.n_layer)
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
            if args.n_embd >= 4096:
                D_MIX_LORA = 64
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            D_DECAY_LORA = 64
            if args.n_embd >= 4096:
                D_DECAY_LORA = 128
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x)
        x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)

        return self.jit_func_2(x, g)

########################################################################################################

class RWKV_CMix_x060(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv
    


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x060(args, layer_id)
        self.ffn = RWKV_CMix_x060(args, layer_id)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)
        
    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)
    


class RWKV_Layer(pl.LightningModule):
    def __init__(self, args,flag=True):
        super().__init__()
        self.args = args
        self.flag = flag
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        # print(args.n_layer)
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optim_groups = [{"params": trainable_params, "weight_decay": self.args.weight_decay}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, x):
        args = self.args
        if args.dropout > 0:
            x = self.drop0(x)
        
        for block in self.blocks:
            if args.grad_cp == 1:
                x = deepspeed.checkpointing.checkpoint(block, x)
            else:
                x = block(x)

        x = self.ln_out(x)

        if self.flag:
            x = self.head(x)

        return x

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits = self(idx)
        loss = F.mse_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

##Not sure below
import torch
import torch.nn as nn

class DownSample(nn.Module):
    def __init__(self, dim):
        """
        3D Down-sampling operation
        Args:
            dim (int): Input channel dimension
        """
        super().__init__()
        self.conv_down = nn.Conv3d(4 * dim, dim, kernel_size=1, stride=1, padding=0) 
        self.norm = nn.LayerNorm(4 * dim) 

    def forward(self, x, T, H, W):
        """
        Args:
            x (torch.Tensor): Input tensor [B, T*H*W, dim]
            T (int): Temporal dimension (time steps)
            H (int): Height
            W (int): Width
        """
        # x: [B, T*H*W, dim] -> [B, T, H, W, dim]
        print(x.shape)
        x = x.reshape(x.shape[0], T, H//2, W//2, -1)
        print(x.shape)
        x = self.norm(x)
        # Reshape to [B, dim, T, H, W] for Conv3d
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv_down(x)
        x = x.reshape(x.shape[0],-1,x.shape[1])
        return x

class UpSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_up = nn.Conv3d(dim, 4 * dim, kernel_size=1, stride=1, padding=0)  
        self.norm = nn.LayerNorm(4 * dim)  

    def forward(self, x, T, H, W):
        """
        Args:
            x (torch.Tensor): Input tensor [B, -1, dim]
            T (int): Temporal dimension (time steps)
            H (int): Height
            W (int): Width
        """
        x = x.reshape(x.shape[0], -1, T, H//2, W//2)

        # Apply 3D convolution
        x = self.conv_up(x)

        # Reshape to [B, T, H, W, 4*dim]
        x = x.permute(0, 2, 3, 4, 1)

        # Apply normalization
        x = self.norm(x)

        # Reshape back to [B, T*H*W, dim]
        x = x.reshape(x.shape[0], T * H * W, -1)
        return x


class PatchEmbedding3D(nn.Module):
    def __init__(self, input_dim, embed_dim, patch_size):
        super(PatchEmbedding3D, self).__init__()
        self.projection = nn.Conv3d(
            in_channels=input_dim,
            out_channels=embed_dim,
            kernel_size=patch_size,  # no overlapping
            stride=patch_size,
            # padding=(15, 0, 0)
        )
        self.flatten = nn.Flatten(start_dim=2)  # Flatten spatial and time dimensions (T, H, W)

    def forward(self, x):  # (B, C, T, H, W)
        x = self.projection(x)  # (B, embed_dim, T', H', W')
        print(x.shape)
        x = self.flatten(x)  # (B, embed_dim, T'*H'*W')
        x = x.transpose(1, 2)  # (B, T'*H'*W', embed_dim)
        return x
    


class PatchRecovery3D(nn.Module):
    def __init__(self, output_dim, embed_dim, patch_size):
        super(PatchRecovery3D, self).__init__()
        self.reconstruction = nn.ConvTranspose3d(
            in_channels=embed_dim,
            out_channels=output_dim,
            kernel_size=patch_size,
            stride=patch_size,

        )

    def forward(self, x, temporal_shape, spatial_shape):
        # Reshape to (B, embed_dim, T', H', W') for ConvTranspose3d
        B, N, C = x.shape
        T, H, W = temporal_shape, *spatial_shape
        x = x.transpose(1, 2).view(B, C, T, H, W)
        return self.reconstruction(x)

class RWKV_Weather(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.layer1 = RWKV_Layer(args)
        self.layer2 = RWKV_Layer(args,flag=False)
        self.layer3 = RWKV_Layer(args,flag=False)
        self.layer4 = RWKV_Layer(args)
        self.transform_shape1 = nn.Linear(args.vocab_size,args.n_embd)
        self.transform_shape2 = nn.Linear(args.vocab_size,args.n_embd)
        self.downsample = DownSample(args.n_embd)
        self.upsample = UpSample(args.n_embd)
        self.patch_embed = PatchEmbedding3D(5,args.n_embd,(2,4,4))###
        self.patch_reovery = PatchRecovery3D(5, args.n_embd,(2,4,4))###
        
        if args.load_model:
            self.load_rwkv_from_pretrained(args.load_model)

    def load_rwkv_from_pretrained(self, path):
        self.rwkv.load_state_dict(torch.load(path, map_location="cpu"))

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
     
    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        name_of_trainable_params = [n for n, p in self.named_parameters() if p.requires_grad]
        rank_zero_info(f"Name of trainable parameters in optimizers: {name_of_trainable_params}")
        rank_zero_info(f"Number of trainable parameters in optimizers: {len(trainable_params)}")
        optim_groups = [{"params": trainable_params, "weight_decay": self.args.weight_decay}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
    
    def forward(self,samples):
        x,y= samples['input'],samples['target']
        
        x = self.patch_embed(x)
        x = self.layer1(x)
        print(x.shape)
        skip = x
        x = self.transform_shape1(x)
        x = self.downsample(x,5,30,30)
        print(x.shape)
        x = self.layer2(x)
        print('layer2',x.shape)
        x= self.layer3(x)
        print('layer3',x.shape)
        print(x.shape)
        x = self.upsample(x,5,30,30)
        x = self.layer4(x)
        x = x + skip
        print(x.shape)
        x = self.transform_shape2(x)
        x = self.patch_reovery(x,5,(30,30))
        return x,y
    
    def weighted_temporal_consistency_loss(self,predict_frames, target_frames, weights):

        """
        Args:
            predict_frames [B, C, T, H, W]
            target_frames  [B, C, T, H, W]
            weights  [T-1]
        """

        assert predict_frames.shape == target_frames.shape
        num_frames = predict_frames.shape[2]

        assert len(weights) == num_frames - 1
        

        pred_diffs = predict_frames[:, :, 1:] - predict_frames[:, :, :-1]  # [B, C, T-1, H, W]
        target_diffs = target_frames[:, :, 1:] - target_frames[:, :, :-1]  # [B, C, T-1, H, W]
        weights = torch.as_tensor(weights, dtype=torch.float32, device=predict_frames.device).view(1, 1, -1, 1, 1)  # [1, 1, T-1, 1, 1]
        
        loss = (weights * F.mse_loss(pred_diffs, target_diffs, reduction='none')).mean()
        return loss
    
    def training_step(self, batch, batch_idx):
        outputs, targets = self(batch)  
        try:
            print(outputs.shape)
        except:
            print(outputs)
        loss1 = F.mse_loss(outputs, targets)
        weights = torch.ones(outputs.shape[2] - 1, device=outputs.device)  ###
        loss2 = self.weighted_temporal_consistency_loss(outputs, targets, weights)###
        loss = loss1 + loss2

        self.log("batch_loss", loss, on_step=True, on_epoch=False, prog_bar=True) 
        return {"loss": loss}

    
    def training_epoch_end(self, outputs):
        all_losses = [x['loss'] for x in outputs if 'loss' in x]
        train_loss = sum(all_losses) / len(all_losses)
        self.log("train_loss", train_loss, sync_dist=True)
        my_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("my_lr", my_lr, sync_dist=True)



    def validation_step(self, batch, batch_idx):
        outputs, targets = self(batch)
        val_loss = F.mse_loss(outputs, targets)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        val_losses = [x['val_loss'] for x in outputs if 'val_loss' in x]
        if val_losses:
            avg_val_loss = sum(val_losses) / len(val_losses)
            self.log("epoch_val_loss", avg_val_loss, sync_dist=True, prog_bar=True)



    @torch.no_grad()
    def predict(self, input) -> list[int]:
        pass



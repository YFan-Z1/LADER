import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )
    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
        identity = x
        x = self.projector(x)
        return identity + x


class Mona(torch.nn.Module):
    def __init__(self, in_dim, hw_shape, dropout=0.1, INNER_DIM=64):
        super().__init__()
        self.hw_shape = hw_shape
        self.project1 = nn.Linear(in_dim, INNER_DIM)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(INNER_DIM, in_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.adapter_conv = MonaOp(INNER_DIM)
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.project1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.project2.weight)
            nn.init.zeros_(self.project1.bias)
            nn.init.zeros_(self.project2.bias)

    def forward(self, x, add_residual=False):
        ### x's shape is (HW+1)BC, batch_first is False
        x = x.permute(1, 0, 2)
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax
        project1 = self.project1(x)
        b, n, c = project1.shape
        h, w = self.hw_shape
        cls_token, project1 = project1[:, 0:1, :], project1[:, 1:, :]
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n - 1, c)
        project1 = torch.cat((cls_token, project1), dim=1)  # b n c
        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)
        if add_residual:
            out = identity + project2
        else:
            out = project2
        return out.permute(1, 0, 2)


class Adapter(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        # _before
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout
        self.init_option = init_option
        self._reset_parameters()

    def _reset_parameters(self):
        if self.init_option == "bert":
            raise NotImplementedError
        elif self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=False, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        if add_residual:
            output = up + residual
        else:
            output = up
        return output

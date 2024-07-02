import torch.nn as nn
import torch
from torch import einsum
from tqdm import tqdm
from einops import rearrange, reduce

from utils.losses import NLLSurvLoss, CoxLoss, SurvRankingLoss
from sksurv.util import Surv

def create_mlp(in_dim=None, hid_dims=[], act=nn.ReLU(), dropout=0.,
               out_dim=None, end_with_fc=True, bias=True):

    layers = []
    if len(hid_dims) < 0:
        mlp = nn.Identity()
    elif len(hid_dims) >= 0:
        if len(hid_dims) > 0:
            for hid_dim in hid_dims:
                layers.append(nn.Linear(in_dim, hid_dim, bias=bias))
                layers.append(act)
                layers.append(nn.Dropout(dropout))
                in_dim = hid_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if not end_with_fc:
            layers.append(act)
        mlp = nn.Sequential(*layers)
    return mlp

#
# Multimodal components (Some of the functions were adapted from SurvPath)
#
class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim * mult)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mult), int(dim * mult))
        )

    def forward(self, x):
        return self.net(self.norm(x))


class FeedForwardEnsemble(nn.Module):
    def __init__(self, dim, mult=1, dropout=0., num=16):
        super().__init__()
        self.num = num
        self.norm = nn.LayerNorm(dim)
        self.net = nn.ModuleList([FeedForward(dim, mult, dropout) for _ in range(num)])

    def forward(self, x):
        """
        Args:
            x: (B, proto, d)
        """
        assert x.shape[1] == self.num
        out = []
        for idx in range(self.num):
            out.append(self.net[idx](x[:,idx:idx+1,:]))
        out = torch.cat(out, dim=1)

        return out

class MMAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            residual=True,
            residual_conv_kernel=33,
            eps=1e-8,
            dropout=0.,
            num_pathways=281,
            attn_mode='full'
    ):
        """

        Args:
            dim:
            dim_head:
            heads:
            residual:
            residual_conv_kernel:
            eps:
            dropout:
            num_pathways:
            attn_mode: ['full', 'partial', 'cross']
                'full': All pairs between P and H
                'partial': P->P, H->P, P->H
                'cross': P->H, H->P
                'self': P->P, H->H
        """
        super().__init__()
        self.num_pathways = num_pathways
        self.eps = eps
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.residual = residual
        self.attn_mode = attn_mode

        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def set_attn_mode(self, attn_mode):
        self.attn_mode = attn_mode

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, eps = *x.shape, self.heads, self.num_pathways, self.eps

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/histology queries and keys
        q_pathways = q[:, :, :self.num_pathways, :]  # bs x head x num_pathways x dim
        k_pathways = k[:, :, :self.num_pathways, :]

        q_histology = q[:, :, self.num_pathways:, :]  # bs x head x num_patches x dim
        k_histology = k[:, :, self.num_pathways:, :]

        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        cross_attn_histology = einsum(einops_eq, q_histology, k_pathways)
        attn_pathways = einsum(einops_eq, q_pathways, k_pathways)
        cross_attn_pathways = einsum(einops_eq, q_pathways, k_histology)
        attn_histology = einsum(einops_eq, q_histology, k_histology)

        # softmax
        pre_softmax_cross_attn_histology = cross_attn_histology
        if self.attn_mode == 'full': # H->P, P->H, P->P, H->H
            cross_attn_histology = cross_attn_histology.softmax(dim=-1)
            attn_histology_pathways = torch.cat((cross_attn_histology, attn_histology), dim=-1).softmax(dim=-1)
            attn_pathways_histology = torch.cat((attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)

            # compute output
            out_pathways = attn_pathways_histology @ v
            out_histology = attn_histology_pathways @ v
        elif self.attn_mode == 'cross': # P->H, H->P
            cross_attn_histology = cross_attn_histology.softmax(dim=-1)
            cross_attn_pathways = cross_attn_pathways.softmax(dim=-1)

            # compute output
            out_pathways = cross_attn_pathways @ v[:, :, self.num_pathways:]
            out_histology = cross_attn_histology @ v[:, :, :self.num_pathways]
        elif self.attn_mode == 'self': # P->P, H->H (Late fusion)
            attn_histology = attn_histology.softmax(dim=-1)
            attn_pathways = attn_pathways.softmax(dim=-1)

            out_pathways = attn_pathways @ v[:, :, :self.num_pathways]
            out_histology = attn_histology @ v[:, :, self.num_pathways:]
        elif self.attn_mode == 'partial': # H->P, P->H, P->P (SURVPATH)
            cross_attn_histology = cross_attn_histology.softmax(dim=-1)
            attn_pathways_histology = torch.cat((attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)

            # compute output
            out_pathways = attn_pathways_histology @ v
            out_histology = cross_attn_histology @ v[:, :, :self.num_pathways]
        elif self.attn_mode == 'mcat': # P->P, P->H
            cross_attn_pathways = cross_attn_pathways.softmax(dim=-1)

            out_pathways = q_pathways
            out_histology = cross_attn_pathways @ v[:, :, self.num_pathways:]
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}")

        out = torch.cat((out_pathways, out_histology), dim=2)

        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)

        if return_attn:
            # return three matrices
            return out, attn_pathways.squeeze().detach().cpu(), cross_attn_pathways.squeeze().detach().cpu(), pre_softmax_cross_attn_histology.squeeze().detach().cpu()

        return out


class MMAttentionLayer(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
            self,
            norm_layer=nn.LayerNorm,
            dim=512,
            dim_head=64,
            heads=6,
            residual=True,
            dropout=0.,
            num_pathways=281,
            attn_mode='full'
    ):

        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways
        self.attn_mode = attn_mode

        self.attn = MMAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways,
            attn_mode=attn_mode
        )

    def set_attn_mode(self, attn_mode):
        self.attn.set_attn_mode(attn_mode)

    def forward(self, x=None, mask=None, return_attention=False):

        if return_attention:
            x, attn_pathways, cross_attn_pathways, cross_attn_histology = self.attn(x=self.norm(x), mask=mask,
                                                                                    return_attn=True)
            return x, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            x = self.attn(x=self.norm(x), mask=mask)

        return x

def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))



#
# Model processing
#
def predict_emb(self, dataset, use_cuda=True, permute=False):
    """
    Create prototype-based slide representation

    Returns
    - X (torch.Tensor): (n_data x output_set_dim)
    - y (torch.Tensor): (n_data)
    """

    X = []

    for i in tqdm(range(len(dataset))):
        batch = dataset.__getitem__(i)
        data = batch['img'].unsqueeze(dim=0)
        if use_cuda:
            data = data.cuda()
        
        with torch.no_grad():
            out = self.representation(data)
            out = out['repr'].data.detach().cpu()

        X.append(out)

    X = torch.cat(X)

    return X

def predict_surv(self, dataset,  use_cuda=True, permute=False):
    """
    Create prototype-based slide representation
    """

    output = []
    label_output = []
    censor_output = []
    time_output = []

    for i in tqdm(range(len(dataset))):
        batch = dataset.__getitem__(i)
        data, label, censorship, time = batch['img'].unsqueeze(dim=0), batch['label'].unsqueeze(dim=0), batch['censorship'].unsqueeze(dim=0), batch['survival_time'].unsqueeze(dim=0)
        batch_size = data.shape[0]

        if use_cuda:
            data = data.cuda()

        with torch.no_grad():
            batch_out = self.representation(data)
            batch_out = batch_out['repr'].data.cpu()

        output.append(batch_out)
        label_output.append(label)
        censor_output.append(censorship)
        time_output.append(time)

    output = torch.cat(output)
    label_output = torch.cat(label_output)
    censor_output = torch.cat(censor_output)
    time_output = torch.cat(time_output)

    y = Surv.from_arrays(~censor_output.numpy().astype('bool').squeeze(),
                            time_output.numpy().squeeze()
                            )
    
    return output, y


def process_surv(logits, label, censorship, loss_fn=None):
    results_dict = {'logits': logits}
    log_dict = {}

    if loss_fn is not None and label is not None:
        if isinstance(loss_fn, NLLSurvLoss):
            surv_loss_dict = loss_fn(logits=logits, times=label, censorships=censorship)
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).unsqueeze(dim=1)
            results_dict.update({'hazards': hazards,
                                    'survival': survival,
                                    'risk': risk})
        elif isinstance(loss_fn, CoxLoss):
            # logits is log risk
            surv_loss_dict = loss_fn(logits=logits, times=label, censorships=censorship)
            risk = torch.exp(logits)
            results_dict['risk'] = risk

        elif isinstance(loss_fn, SurvRankingLoss):
                surv_loss_dict = loss_fn(z=logits, times=label, censorships=censorship)
                results_dict['risk'] = logits

        loss = surv_loss_dict['loss']
        log_dict['surv_loss'] = surv_loss_dict['loss'].item()
        log_dict.update(
            {k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)})
        results_dict.update({'loss': loss})

    return results_dict, log_dict

def predict_clf(self, dataset, use_cuda=True, permute=False):
    """
    Create prototype-based slide representation

    Returns
    - X (torch.Tensor): (n_data x output_set_dim)
    - y (torch.Tensor): (n_data)
    """

    X, y = [], []

    for i in tqdm(range(len(dataset))):
        batch = dataset.__getitem__(i)
        data = batch['img'].unsqueeze(dim=0)
        label = batch['label']
        if use_cuda:
            data = data.cuda()
        
        with torch.no_grad():
            out = self.representation(data)
            out = out['repr'].data.detach().cpu()

        X.append(out)
        y.append(label)

    X = torch.cat(X)
    y = torch.Tensor(y).to(torch.long)

    return X, y

def process_clf(logits, label, loss_fn):
    results_dict = {'logits': logits}
    log_dict = {}

    if loss_fn is not None and label is not None:
        cls_loss = loss_fn(logits, label)
        loss = cls_loss
        log_dict.update({
            'cls_loss': cls_loss.item(),
            'loss': loss.item()})
        results_dict.update({'loss': loss})
    
    return results_dict, log_dict


#
# Attention networks
#
class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(D, n_classes)]

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout)]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid(),
                            nn.Dropout(dropout)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A

"""
Causal-Confounding Prototype Separation Model for Multimodal Survival Prediction

This module implements a three-branch architecture that separates prototypes into
causal and confounding groups, allowing for disentangled representation learning.

Architecture:
    Input Layer:
    ├─ Histology: 16 prototype representations {z_h,1, ..., z_h,16}
    └─ Genomics: 50 pathway representations {z_g,1, ..., z_g,50}

    Prototype Selection Layer:
    ├─ Histology Selection Network G_h: learns selection weights w_h ∈ [0,1]^16
    │   └─ Output: α_h,i = σ(MLP(z_h,i)) indicates probability that prototype i is causal
    ├─ Genomics Selection Network G_g: learns selection weights w_g ∈ [0,1]^50
    │   └─ Output: α_g,j = σ(MLP(z_g,j)) indicates probability that pathway j is causal
    └─ Uses Gumbel-Softmax or Top-K for differentiable discrete selection

    Prototype Separation Layer:
    ├─ Causal prototypes: z^c_h = {z_h,i | α_h,i > τ}, z^c_g = {z_g,j | α_g,j > τ}
    └─ Confounding prototypes: z^s_h = {z_h,i | α_h,i ≤ τ}, z^s_g = {z_g,j | α_g,j ≤ τ}

    Three-branch Prediction Network:
    ├─ Causal Branch F_causal:
    │   ├─ Input: z^c_h and z^c_g (only causal prototypes)
    │   ├─ Transformer fusion
    │   └─ Output: risk prediction ŷ_causal
    │
    ├─ Confounding Branch F_confound:
    │   ├─ Input: z^s_h and z^s_g (only confounding prototypes)
    │   ├─ Transformer fusion
    │   └─ Output: risk prediction ŷ_confound
    │
    └─ Full Branch F_full:
        ├─ Input: all prototypes
        ├─ Transformer fusion (similar to original MMP)
        └─ Output: risk prediction ŷ_full

Author: Claude
Date: 2025-10-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .prototype_selection import (
    PrototypeSelectionNetwork,
    TopKPrototypeSelector,
    AdaptivePrototypeSelector
)
from .components import (
    SNN_Block,
    MMAttentionLayer,
    FeedForward,
    FeedForwardEnsemble,
    process_surv
)


def init_per_path_model(omic_sizes, hidden_dim=256):
    """
    Create a list of SNNs, one for each pathway

    Args:
        omic_sizes: List of integers, each indicating number of genes per prototype
    """
    hidden = [hidden_dim, hidden_dim]
    sig_networks = []
    for input_dim in omic_sizes:
        fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
        sig_networks.append(nn.Sequential(*fc_omic))
    sig_networks = nn.ModuleList(sig_networks)

    return sig_networks


def agg_histo(X, agg_mode='mean'):
    """
    Aggregating histology prototypes
    """
    if agg_mode == 'mean':
        out = torch.mean(X, dim=1)
    elif agg_mode == 'cat':
        out = X.reshape(X.shape[0], -1)
    else:
        raise NotImplementedError(f"Not implemented for {agg_mode}")

    return out


def construct_proto_embedding(path_proj_dim, append_embed='modality', numOfproto_histo=16, numOfproto_omics=50):
    """
    Per-prototype learnable/non-learnable embeddings to append to the original prototype embeddings
    """
    if append_embed == 'modality':  # One-hot encoding for two modalities
        path_proj_dim_new = path_proj_dim + 2

        histo_embedding = torch.tensor([[[1, 0]]]).repeat(1, numOfproto_histo, 1)
        gene_embedding = torch.tensor([[[0, 1]]]).repeat(1, numOfproto_omics, 1)

    elif append_embed == 'proto':
        path_proj_dim_new = path_proj_dim + numOfproto_histo + numOfproto_omics
        embedding = torch.eye(numOfproto_histo + numOfproto_omics).unsqueeze(0)

        histo_embedding = embedding[:, :numOfproto_histo, :]
        gene_embedding = embedding[:, numOfproto_histo:, :]

    elif append_embed == 'random':
        append_dim = 32
        path_proj_dim_new = path_proj_dim + append_dim

        histo_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_histo, append_dim), requires_grad=True)
        gene_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_omics, append_dim), requires_grad=True)

    else:
        path_proj_dim_new = path_proj_dim
        histo_embedding = None
        gene_embedding = None

    return path_proj_dim_new, histo_embedding, gene_embedding


class SharedTransformerBranch(nn.Module):
    """
    A single branch of the three-branch architecture.

    This is a simplified version that uses standard self-attention instead of
    the custom MMAttentionLayer, making it compatible with dynamic prototype numbers.
    """

    def __init__(self,
                 path_proj_dim=256,
                 num_coattn_layers=1,
                 dropout=0.1,
                 mult=1,
                 histo_agg='mean'):
        super().__init__()

        self.path_proj_dim = path_proj_dim
        self.num_coattn_layers = num_coattn_layers
        self.histo_agg = histo_agg
        self.out_mult = mult

        if num_coattn_layers == 0:
            # Simple feedforward without attention
            out_dim = path_proj_dim
            self.coattn = nn.Sequential(
                FeedForward(out_dim, mult, dropout=dropout),
                nn.LayerNorm(int(out_dim * mult))
            )
            self.out_dim = int(out_dim * mult)
        else:
            # Use standard Transformer layer with self-attention
            # This works with dynamic number of tokens
            out_dim = path_proj_dim // 2

            # Standard PyTorch MultiheadAttention
            self.self_attn = nn.MultiheadAttention(
                embed_dim=path_proj_dim,
                num_heads=1,
                dropout=dropout,
                batch_first=True
            )

            self.norm1 = nn.LayerNorm(path_proj_dim)
            self.norm2 = nn.LayerNorm(int(out_dim * mult))

            # Project to output dimension
            self.proj = nn.Linear(path_proj_dim, out_dim)

            # Feedforward
            self.feed_forward = FeedForward(out_dim, mult, dropout=dropout)

            self.out_dim = int(out_dim * mult)

    def forward(self, h_omic, h_path):
        """
        Args:
            h_omic: (B, n_gene_proto, d) - Gene pathway embeddings
            h_path: (B, n_histo_proto, d) - Histology prototype embeddings

        Returns:
            embedding: (B, out_dim) - Fused embedding
        """
        # Concatenate gene and histo prototypes
        tokens = torch.cat([h_omic, h_path], dim=1)  # (B, n_gene + n_histo, d)

        num_pathways = h_omic.shape[1]

        if self.num_coattn_layers > 0:
            # Apply self-attention
            # PyTorch MultiheadAttention expects (B, N, D) with batch_first=True
            attn_out, _ = self.self_attn(tokens, tokens, tokens)
            tokens = self.norm1(tokens + attn_out)  # Residual connection

            # Project and apply feedforward
            tokens = self.proj(tokens)  # (B, n_gene + n_histo, out_dim)
            ff_out = self.feed_forward(tokens)
            mm_embed = self.norm2(ff_out)
        else:
            # Just feedforward
            mm_embed = self.coattn(tokens)

        # Aggregate
        # Gene aggregation
        gene_embed = mm_embed[:, :num_pathways, :]
        gene_embed = torch.mean(gene_embed, dim=1)  # (B, out_dim)

        # Histo aggregation
        histo_embed = mm_embed[:, num_pathways:, :]
        histo_embed = agg_histo(histo_embed, self.histo_agg)  # (B, out_dim)

        # Combine both modalities
        embedding = torch.cat([gene_embed, histo_embed], dim=1)  # (B, 2*out_dim)

        return embedding


class CausalSeparationModel(nn.Module):
    """
    Main model with three-branch architecture for causal-confounding separation.
    """

    def __init__(self,
                 omic_sizes=[100, 200, 300],
                 histo_in_dim=1024,
                 dropout=0.1,
                 num_classes=4,
                 path_proj_dim=256,
                 num_coattn_layers=1,
                 histo_agg='mean',
                 histo_model='PANTHER',
                 append_embed='none',
                 mult=1,
                 numOfproto=16,
                 selection_method='gumbel',  # 'gumbel', 'topk', 'adaptive'
                 tau=1.0,
                 hard=True,
                 top_k_ratio=0.5):
        """
        Args:
            omic_sizes: List of integers, each indicating number of genes per prototype
            histo_in_dim: Dimension of histology feature embedding (for PANTHER: prob+mean+cov)
            num_classes: 4 if using NLL, 1 if using Cox/Ranking loss
            path_proj_dim: Dimension of the embedding space for fusion
            num_coattn_layers: Number of co-attention layers
            histo_agg: 'mean' or 'cat' for aggregating histology prototypes
            histo_model: 'PANTHER', 'OT', 'H2T', etc.
            append_embed: Type of positional embedding to append
            mult: Multiplier for feedforward dimension
            numOfproto: Number of histology prototypes
            selection_method: 'gumbel', 'topk', or 'adaptive'
            tau: Temperature for Gumbel-Softmax
            hard: Use hard Gumbel-Softmax
            top_k_ratio: Ratio for top-k selection
        """
        super().__init__()

        self.num_pathways = len(omic_sizes)
        self.numOfproto = numOfproto
        self.num_classes = num_classes
        self.histo_model = histo_model.lower()
        self.histo_agg = histo_agg
        self.append_embed = append_embed
        self.selection_method = selection_method
        self.out_mult = mult

        # Gene pathway networks (shared across all branches)
        self.sig_networks = init_per_path_model(omic_sizes, hidden_dim=256)

        # Histology projection (shared across all branches)
        if self.histo_model == 'panther':  # Uses prob/mean/cov
            self.path_proj_net = nn.Sequential(nn.Linear(histo_in_dim * 2 + 1, path_proj_dim))
        else:
            self.path_proj_net = nn.Sequential(nn.Linear(histo_in_dim, path_proj_dim))

        # Prototype embeddings
        if self.histo_model != "mil":
            self.path_proj_dim, self.histo_embedding, self.gene_embedding = construct_proto_embedding(
                path_proj_dim,
                self.append_embed,
                self.numOfproto,
                len(omic_sizes)
            )
        else:
            self.path_proj_dim = path_proj_dim
            self.histo_embedding = None
            self.gene_embedding = None

        # Prototype selection networks
        if selection_method == 'gumbel':
            self.histo_selector = PrototypeSelectionNetwork(
                proto_dim=self.path_proj_dim,
                hidden_dim=256,
                tau=tau,
                hard=hard,
                use_gumbel=True
            )
            self.gene_selector = PrototypeSelectionNetwork(
                proto_dim=self.path_proj_dim,
                hidden_dim=256,
                tau=tau,
                hard=hard,
                use_gumbel=True
            )
        elif selection_method == 'topk':
            self.histo_selector = TopKPrototypeSelector(
                proto_dim=self.path_proj_dim,
                hidden_dim=256,
                top_k_ratio=top_k_ratio
            )
            self.gene_selector = TopKPrototypeSelector(
                proto_dim=self.path_proj_dim,
                hidden_dim=256,
                top_k_ratio=top_k_ratio
            )
        elif selection_method == 'adaptive':
            self.histo_selector = AdaptivePrototypeSelector(
                proto_dim=self.path_proj_dim,
                hidden_dim=256
            )
            self.gene_selector = AdaptivePrototypeSelector(
                proto_dim=self.path_proj_dim,
                hidden_dim=256
            )
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")

        # Three branches (shared architecture)
        self.causal_branch = SharedTransformerBranch(
            path_proj_dim=self.path_proj_dim,
            num_coattn_layers=num_coattn_layers,
            dropout=dropout,
            mult=mult,
            histo_agg=histo_agg
        )

        self.confound_branch = SharedTransformerBranch(
            path_proj_dim=self.path_proj_dim,
            num_coattn_layers=num_coattn_layers,
            dropout=dropout,
            mult=mult,
            histo_agg=histo_agg
        )

        self.full_branch = SharedTransformerBranch(
            path_proj_dim=self.path_proj_dim,
            num_coattn_layers=num_coattn_layers,
            dropout=dropout,
            mult=mult,
            histo_agg=histo_agg
        )

        # Calculate output dimensions
        branch_out_dim = self.causal_branch.out_dim * 2  # *2 because of gene+histo concatenation

        # Classifiers for each branch
        self.causal_classifier = nn.Linear(branch_out_dim, num_classes, bias=False)
        self.confound_classifier = nn.Linear(branch_out_dim, num_classes, bias=False)
        self.full_classifier = nn.Linear(branch_out_dim, num_classes, bias=False)

    def forward_no_loss(self, x_path, x_omics, return_selection=False):
        """
        Forward pass without computing loss.

        Args:
            x_path: (B, numOfproto, in_dim) - Histology prototypes
            x_omics: List of (B, n_genes) tensors - Gene expressions per pathway
            return_selection: Whether to return selection masks

        Returns:
            dict with keys:
                - 'causal_logits': (B, num_classes)
                - 'confound_logits': (B, num_classes)
                - 'full_logits': (B, num_classes)
                - 'histo_causal_mask': (B, numOfproto) if return_selection
                - 'gene_causal_mask': (B, num_pathways) if return_selection
        """
        device = x_path.device
        B = x_path.shape[0]

        # ====== Gene pathway embeddings ======
        h_omic = []
        for idx, sig_feat in enumerate(x_omics):
            omic_feat = self.sig_networks[idx](sig_feat.float())  # (B, d)
            h_omic.append(omic_feat)
        h_omic = torch.stack(h_omic, dim=1)  # (B, num_pathways, d)

        if self.gene_embedding is not None:
            arr = []
            for idx in range(len(h_omic)):
                arr.append(torch.cat([h_omic[idx:idx + 1], self.gene_embedding.to(device)], dim=-1))
            h_omic = torch.cat(arr, dim=0)

        # ====== Histology embeddings ======
        h_path = self.path_proj_net(x_path)  # (B, numOfproto, path_proj_dim)

        if self.histo_embedding is not None:
            arr = []
            for idx in range(len(h_path)):
                arr.append(torch.cat([h_path[idx:idx + 1], self.histo_embedding.to(device)], dim=-1))
            h_path = torch.cat(arr, dim=0)

        # ====== Prototype selection ======
        histo_selection = self.histo_selector(h_path, return_scores=True)
        gene_selection = self.gene_selector(h_omic, return_scores=True)

        # Extract masks
        histo_causal_mask = histo_selection['causal_mask']      # (B, numOfproto)
        histo_confound_mask = histo_selection['confound_mask']  # (B, numOfproto)
        gene_causal_mask = gene_selection['causal_mask']        # (B, num_pathways)
        gene_confound_mask = gene_selection['confound_mask']    # (B, num_pathways)

        # ====== Apply masks to separate causal/confounding prototypes ======
        # Expand masks for element-wise multiplication
        histo_causal_mask_expanded = histo_causal_mask.unsqueeze(-1)      # (B, numOfproto, 1)
        histo_confound_mask_expanded = histo_confound_mask.unsqueeze(-1)
        gene_causal_mask_expanded = gene_causal_mask.unsqueeze(-1)        # (B, num_pathways, 1)
        gene_confound_mask_expanded = gene_confound_mask.unsqueeze(-1)

        # Soft masking: multiply features by selection weights
        h_path_causal = h_path * histo_causal_mask_expanded
        h_path_confound = h_path * histo_confound_mask_expanded
        h_omic_causal = h_omic * gene_causal_mask_expanded
        h_omic_confound = h_omic * gene_confound_mask_expanded

        # ====== Three-branch forward ======
        # Causal branch: only causal prototypes
        causal_embedding = self.causal_branch(h_omic_causal, h_path_causal)
        causal_logits = self.causal_classifier(causal_embedding)

        # Confounding branch: only confounding prototypes
        confound_embedding = self.confound_branch(h_omic_confound, h_path_confound)
        confound_logits = self.confound_classifier(confound_embedding)

        # Full branch: all prototypes
        full_embedding = self.full_branch(h_omic, h_path)
        full_logits = self.full_classifier(full_embedding)

        # ====== Prepare outputs ======
        out = {
            'causal_logits': causal_logits,
            'confound_logits': confound_logits,
            'full_logits': full_logits,
            'causal_embedding': causal_embedding,
            'confound_embedding': confound_embedding,
            'full_embedding': full_embedding,
        }

        if return_selection:
            out['histo_causal_mask'] = histo_causal_mask
            out['histo_confound_mask'] = histo_confound_mask
            out['gene_causal_mask'] = gene_causal_mask
            out['gene_confound_mask'] = gene_confound_mask
            out['histo_selection'] = histo_selection
            out['gene_selection'] = gene_selection

        return out

    def forward(self, x_path, x_omics, label=None, censorship=None, loss_fn=None, return_selection=False):
        """
        Full forward pass with loss computation.

        Args:
            x_path: (B, numOfproto, in_dim)
            x_omics: List of (B, n_genes) tensors
            label: Survival labels
            censorship: Censorship indicators
            loss_fn: Loss function name ('cox', 'nll', etc.)
            return_selection: Whether to return selection information

        Returns:
            results_dict: Dictionary with predictions and losses
            log_dict: Dictionary with logging information
        """
        # Forward pass without loss
        out = self.forward_no_loss(x_path, x_omics, return_selection=return_selection)

        # Compute losses for each branch
        causal_results, causal_log = process_surv(
            out['causal_logits'], label, censorship, loss_fn
        )
        confound_results, confound_log = process_surv(
            out['confound_logits'], label, censorship, loss_fn
        )
        full_results, full_log = process_surv(
            out['full_logits'], label, censorship, loss_fn
        )

        # Prepare output dictionaries
        results_dict = {
            # Logits from each branch
            'causal_logits': out['causal_logits'],
            'confound_logits': out['confound_logits'],
            'full_logits': out['full_logits'],
            'logits': out['full_logits'],  # Default to full branch for compatibility

            # Embeddings
            'causal_embedding': out['causal_embedding'],
            'confound_embedding': out['confound_embedding'],
            'full_embedding': out['full_embedding'],

            # Losses
            'causal_loss': causal_results.get('loss', None),
            'confound_loss': confound_results.get('loss', None),
            'full_loss': full_results.get('loss', None),

            # Risks (for C-index computation)
            'causal_risk': causal_results.get('risk', None),
            'confound_risk': confound_results.get('risk', None),
            'full_risk': full_results.get('risk', None),
            'risk': full_results.get('risk', None),  # Default to full branch
        }

        if return_selection:
            results_dict.update({
                'histo_causal_mask': out['histo_causal_mask'],
                'histo_confound_mask': out['histo_confound_mask'],
                'gene_causal_mask': out['gene_causal_mask'],
                'gene_confound_mask': out['gene_confound_mask'],
                'histo_selection': out['histo_selection'],
                'gene_selection': out['gene_selection'],
            })

        log_dict = {
            'causal_loss': causal_log.get('loss', 0.0),
            'confound_loss': confound_log.get('loss', 0.0),
            'full_loss': full_log.get('loss', 0.0),
        }

        return results_dict, log_dict


if __name__ == '__main__':
    """
    Test the causal separation model
    """
    print("=" * 80)
    print("Testing Causal Separation Model")
    print("=" * 80)

    # Test configuration
    batch_size = 4
    n_proto_histo = 16
    n_pathways = 50
    histo_in_dim = 1024

    # Create dummy data
    # For PANTHER: each prototype has [prob, mean, cov]
    # Assuming proto_dim = 512, so total = 1 + 512 + 512 = 1025 per prototype
    x_path = torch.randn(batch_size, n_proto_histo, 1025)

    # Gene data: list of tensors, one per pathway
    omic_sizes = [281] * n_pathways  # 50 pathways, each with 281 genes
    x_omics = [torch.randn(batch_size, size) for size in omic_sizes]

    # Dummy labels
    label = torch.randint(0, 4, (batch_size,))
    censorship = torch.randint(0, 2, (batch_size,))

    # Create model
    model = CausalSeparationModel(
        omic_sizes=omic_sizes,
        histo_in_dim=1025,
        num_classes=4,
        path_proj_dim=256,
        num_coattn_layers=1,
        histo_model='PANTHER',
        numOfproto=n_proto_histo,
        selection_method='gumbel',
        tau=1.0,
        hard=True
    )

    print(f"\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\n" + "-" * 80)
    print("Testing forward pass...")

    results, logs = model(
        x_path, x_omics,
        label=label,
        censorship=censorship,
        loss_fn='nll',
        return_selection=True
    )

    print(f"\nOutput shapes:")
    print(f"  Causal logits: {results['causal_logits'].shape}")
    print(f"  Confound logits: {results['confound_logits'].shape}")
    print(f"  Full logits: {results['full_logits'].shape}")

    print(f"\nSelection info:")
    print(f"  Histo causal mask: {results['histo_causal_mask'].shape}")
    print(f"  Gene causal mask: {results['gene_causal_mask'].shape}")
    print(f"  Avg causal histo prototypes: {results['histo_causal_mask'].sum(1).mean():.2f}")
    print(f"  Avg causal gene prototypes: {results['gene_causal_mask'].sum(1).mean():.2f}")

    print(f"\nLosses:")
    print(f"  Causal loss: {logs['causal_loss']:.4f}")
    print(f"  Confound loss: {logs['confound_loss']:.4f}")
    print(f"  Full loss: {logs['full_loss']:.4f}")

    # Test backward
    print("\n" + "-" * 80)
    print("Testing backward pass...")

    total_loss = results['causal_loss'] + results['confound_loss'] + results['full_loss']
    total_loss.backward()

    print("Backward pass successful!")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)

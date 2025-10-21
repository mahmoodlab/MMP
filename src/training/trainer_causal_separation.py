"""
Trainer for Causal-Confounding Separation Model

This module extends the standard trainer to support the three-branch
causal separation architecture with specialized loss functions.

Author: Claude
Date: 2025-10-21
"""

import os
from os.path import join as j_
import pdb
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn

try:
    from sksurv.metrics import concordance_index_censored
except ImportError:
    print('scikit-survival not installed. Exiting...')
    raise

from mil_models.tokenizer import PrototypeTokenizer
from mil_models import create_multimodal_survival_model, prepare_emb
from mil_models.causal_losses import CausalSeparationLoss, compute_causal_separation_metrics
from utils.losses import NLLSurvLoss, CoxLoss, SurvRankingLoss
from utils.utils import (EarlyStopping, save_checkpoint, AverageMeter, safe_list_to,
                         get_optim, print_network, get_lr_scheduler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROTO_MODELS = ['PANTHER', 'OT', 'H2T', 'ProtoCount']


def train_causal_separation(datasets, args):
    """
    Train the causal separation model for a single fold.

    Args:
        datasets: Dictionary of DataLoaders for 'train', 'val', 'test'
        args: Training arguments

    Returns:
        results: Dictionary of evaluation results per split
        dumps: Dictionary of detailed outputs per split
    """

    writer_dir = args.results_dir
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    assert args.es_metric == 'loss'

    # Initialize base survival loss
    if args.loss_fn == 'nll':
        base_loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
    elif args.loss_fn == 'cox':
        base_loss_fn = CoxLoss()
    elif args.loss_fn == 'rank':
        base_loss_fn = SurvRankingLoss()

    # Initialize causal separation loss
    causal_loss_fn = CausalSeparationLoss(
        survival_loss_fn=args.loss_fn,
        lambda_causal_consistency=getattr(args, 'lambda_causal_consistency', 1.0),
        lambda_confound_random=getattr(args, 'lambda_confound_random', 0.5),
        lambda_sparsity=getattr(args, 'lambda_sparsity', 0.01),
        lambda_balance=getattr(args, 'lambda_balance', 0.1),
        causal_consistency_mode=getattr(args, 'causal_consistency_mode', 'kl'),
        confound_random_mode=getattr(args, 'confound_random_mode', 'entropy'),
        target_selection_ratio=getattr(args, 'target_selection_ratio', 0.5),
        num_classes=args.n_label_bins if args.loss_fn == 'nll' else 1
    )

    args.feat_dim = args.in_dim  # Patch feature dimension
    print('\nInit Causal Separation Model...', end=' ')

    # If prototype-based models, need to create slide-level embeddings
    if args.model_histo_type in PROTO_MODELS:
        datasets, _ = prepare_emb(datasets, args, mode='survival')

        new_in_dim = None
        for k, loader in datasets.items():
            assert loader.dataset.X is not None
            new_in_dim_curr = loader.dataset.X.shape[-1]
            if new_in_dim is None:
                new_in_dim = new_in_dim_curr
            else:
                assert new_in_dim == new_in_dim_curr

            # The original embedding is 1-D (long) feature vector
            # Reshape it to (n_proto, -1)
            tokenizer = PrototypeTokenizer(args.model_histo_type, args.out_type, args.n_proto)
            prob, mean, cov = tokenizer(loader.dataset.X)
            loader.dataset.X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1),
                                         torch.Tensor(mean),
                                         torch.Tensor(cov)], dim=-1)

            factor = args.n_proto

        args.in_dim = new_in_dim // factor
    else:
        print(f"{args.model_histo_type} doesn't construct unsupervised slide-level embeddings!")

    ## Set the dimensionality for different inputs
    args.omic_dim = datasets['train'].dataset.omics_data.shape[1]

    if args.omics_modality in ['pathway', 'functional']:
        omic_sizes = datasets['train'].dataset.omic_sizes
    else:
        omic_sizes = []

    model = create_multimodal_survival_model(args, omic_sizes=omic_sizes)
    model.to(device)

    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model=model, args=args)
    lr_scheduler = get_lr_scheduler(args, optimizer, datasets['train'])

    if args.early_stopping:
        print('\nSetup EarlyStopping...', end=' ')
        early_stopper = EarlyStopping(save_dir=args.results_dir,
                                      patience=args.es_patience,
                                      min_stop_epoch=args.es_min_epochs,
                                      better='min' if args.es_metric == 'loss' else 'max',
                                      verbose=True)
    else:
        print('\nNo EarlyStopping...', end=' ')
        early_stopper = None

    #####################
    # The training loop #
    #####################
    for epoch in range(args.max_epochs):
        step_log = {'epoch': epoch, 'samples_seen': (epoch + 1) * len(datasets['train'].dataset)}

        ### Train Loop
        print('#' * 10, f'TRAIN Epoch: {epoch}', '#' * 10)
        train_results = train_loop_causal_separation(
            model, datasets['train'], optimizer, lr_scheduler,
            base_loss_fn, causal_loss_fn,
            print_every=args.print_every,
            accum_steps=args.accum_steps
        )

        ### Validation Loop (Optional)
        if 'val' in datasets.keys():
            print('#' * 11, f'VAL Epoch: {epoch}', '#' * 11)
            val_results, _ = validate_causal_separation(
                model, datasets['val'],
                base_loss_fn, causal_loss_fn,
                print_every=args.print_every,
                verbose=True
            )

            ### Check Early Stopping (Optional)
            if early_stopper is not None:
                if args.es_metric == 'loss':
                    score = val_results['loss']
                else:
                    raise NotImplementedError

                save_ckpt_kwargs = dict(config=vars(args),
                                        epoch=epoch,
                                        model=model,
                                        score=score,
                                        fname=f's_checkpoint.pth')
                stop = early_stopper(epoch, score, save_checkpoint, save_ckpt_kwargs)
                if stop:
                    break
        print('#' * (22 + len(f'TRAIN Epoch: {epoch}')), '\n')

    ### End of epoch: Load in the best model (or save the latest model with no early stopping)
    if args.early_stopping:
        model.load_state_dict(torch.load(j_(args.results_dir, f"s_checkpoint.pth"))['model'])
    else:
        torch.save(model.state_dict(), j_(args.results_dir, f"s_checkpoint.pth"))

    ### End of epoch: Evaluate on val and test set
    results, dumps = {}, {}
    for k, loader in datasets.items():
        print(f'End of training. Evaluating on Split {k.upper()}...:')
        results[k], dumps[k] = validate_causal_separation(
            model, loader,
            base_loss_fn, causal_loss_fn,
            print_every=args.print_every,
            dump_results=True,
            verbose=False
        )

        if k == 'train':
            _ = results.pop('train')

    return results, dumps


def train_loop_causal_separation(model, loader, optimizer, lr_scheduler,
                                 base_loss_fn, causal_loss_fn,
                                 print_every=50, accum_steps=32):
    """
    Training loop for one epoch with causal separation.

    Args:
        model: CausalSeparationModel instance
        loader: DataLoader
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        base_loss_fn: Base survival loss (Cox, NLL, etc.)
        causal_loss_fn: CausalSeparationLoss instance
        print_every: Print interval
        accum_steps: Gradient accumulation steps

    Returns:
        results_dict: Dictionary with training metrics
    """

    model.train()
    meters = {
        'bag_size': AverageMeter(),
        'total_loss': AverageMeter(),
        'survival_loss_causal': AverageMeter(),
        'survival_loss_confound': AverageMeter(),
        'survival_loss_full': AverageMeter(),
        'causal_consistency': AverageMeter(),
        'confound_random': AverageMeter(),
        'sparsity': AverageMeter(),
        'balance': AverageMeter(),
    }

    all_risk_scores, all_censorships, all_event_times = [], [], []

    for batch_idx, batch in enumerate(loader):
        data = safe_list_to(batch['img'], device)
        label = safe_list_to(batch['label'], device)

        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)

        omics = safe_list_to(batch['omics'], device)

        # Forward pass with selection information
        out, log_dict = model(data, omics,
                             label=label,
                             censorship=censorship,
                             loss_fn=base_loss_fn,
                             return_selection=True)

        # Compute combined loss
        total_loss, loss_components = causal_loss_fn(out)

        if total_loss is None or torch.isnan(total_loss):
            continue

        # Backprop with gradient accumulation
        loss = total_loss / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Log survival metrics (using full branch by default)
        all_risk_scores.append(out['risk'].detach().cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())

        # Update meters
        for key, val in loss_components.items():
            if key in meters:
                meters[key].update(val, n=len(data))

        meters['bag_size'].update(data.size(1), n=len(data))

        # Print progress
        if (batch_idx + 1) % print_every == 0:
            print(f'Batch {batch_idx + 1}/{len(loader)} | '
                  f'Loss: {meters["total_loss"].avg:.4f} | '
                  f'Causal: {meters["survival_loss_causal"].avg:.4f} | '
                  f'Confound: {meters["survival_loss_confound"].avg:.4f} | '
                  f'Full: {meters["survival_loss_full"].avg:.4f}')

    # Compute C-index
    all_risk_scores = np.concatenate(all_risk_scores).squeeze()
    all_censorships = np.concatenate(all_censorships).squeeze()
    all_event_times = np.concatenate(all_event_times).squeeze()

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool),
        all_event_times,
        all_risk_scores,
        tied_tol=1e-08
    )[0]

    # Prepare results
    results_dict = {
        'loss': meters['total_loss'].avg,
        'c_index': c_index,
    }

    # Add loss components
    for key in ['survival_loss_causal', 'survival_loss_confound', 'survival_loss_full',
                'causal_consistency', 'confound_random', 'sparsity', 'balance']:
        if key in meters:
            results_dict[key] = meters[key].avg

    print(f'\nTrain Results: Loss={results_dict["loss"]:.4f}, C-Index={c_index:.4f}')

    return results_dict


def validate_causal_separation(model, loader, base_loss_fn, causal_loss_fn,
                               print_every=50, dump_results=False, verbose=True):
    """
    Validation loop for causal separation model.

    Args:
        model: CausalSeparationModel instance
        loader: DataLoader
        base_loss_fn: Base survival loss
        causal_loss_fn: CausalSeparationLoss instance
        print_every: Print interval
        dump_results: Whether to dump detailed results
        verbose: Print verbose output

    Returns:
        results_dict: Dictionary with validation metrics
        dumps_dict: Dictionary with detailed outputs (if dump_results=True)
    """

    model.eval()
    meters = {
        'bag_size': AverageMeter(),
        'total_loss': AverageMeter(),
        'survival_loss_causal': AverageMeter(),
        'survival_loss_confound': AverageMeter(),
        'survival_loss_full': AverageMeter(),
        'causal_consistency': AverageMeter(),
        'confound_random': AverageMeter(),
        'sparsity': AverageMeter(),
        'balance': AverageMeter(),
        'histo_causal_count': AverageMeter(),
        'gene_causal_count': AverageMeter(),
    }

    all_risk_scores = {'causal': [], 'confound': [], 'full': []}
    all_censorships, all_event_times = [], []
    all_case_ids = []

    dumps = {
        'case_id': [],
        'risk_causal': [],
        'risk_confound': [],
        'risk_full': [],
        'censorship': [],
        'survival_time': [],
        'histo_causal_mask': [],
        'gene_causal_mask': [],
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            data = safe_list_to(batch['img'], device)
            label = safe_list_to(batch['label'], device)

            event_time = batch['survival_time'].to(device)
            censorship = batch['censorship'].to(device)

            omics = safe_list_to(batch['omics'], device)

            # Forward pass
            out, log_dict = model(data, omics,
                                 label=label,
                                 censorship=censorship,
                                 loss_fn=base_loss_fn,
                                 return_selection=True)

            # Compute loss
            total_loss, loss_components = causal_loss_fn(out)

            if total_loss is None or torch.isnan(total_loss):
                continue

            # Collect metrics
            all_risk_scores['causal'].append(out['causal_risk'].detach().cpu().numpy())
            all_risk_scores['confound'].append(out['confound_risk'].detach().cpu().numpy())
            all_risk_scores['full'].append(out['full_risk'].detach().cpu().numpy())
            all_censorships.append(censorship.cpu().numpy())
            all_event_times.append(event_time.cpu().numpy())

            # Compute selection metrics
            metrics = compute_causal_separation_metrics(out)

            # Update meters
            for key, val in loss_components.items():
                if key in meters:
                    meters[key].update(val, n=len(data))

            meters['bag_size'].update(data.size(1), n=len(data))
            meters['histo_causal_count'].update(metrics['histo_causal_count'], n=len(data))
            meters['gene_causal_count'].update(metrics['gene_causal_count'], n=len(data))

            # Dump results if requested
            if dump_results:
                dumps['case_id'].extend(batch.get('case_id', [f'sample_{i}' for i in range(len(data))]))
                dumps['risk_causal'].extend(out['causal_risk'].cpu().numpy().tolist())
                dumps['risk_confound'].extend(out['confound_risk'].cpu().numpy().tolist())
                dumps['risk_full'].extend(out['full_risk'].cpu().numpy().tolist())
                dumps['censorship'].extend(censorship.cpu().numpy().tolist())
                dumps['survival_time'].extend(event_time.cpu().numpy().tolist())
                dumps['histo_causal_mask'].extend(out['histo_causal_mask'].cpu().numpy().tolist())
                dumps['gene_causal_mask'].extend(out['gene_causal_mask'].cpu().numpy().tolist())

    # Compute C-indices for all branches
    c_indices = {}
    for branch in ['causal', 'confound', 'full']:
        risk_scores = np.concatenate(all_risk_scores[branch]).squeeze()
        censorships = np.concatenate(all_censorships).squeeze()
        event_times = np.concatenate(all_event_times).squeeze()

        c_index = concordance_index_censored(
            (1 - censorships).astype(bool),
            event_times,
            risk_scores,
            tied_tol=1e-08
        )[0]
        c_indices[f'c_index_{branch}'] = c_index

    # Prepare results
    results_dict = {
        'loss': meters['total_loss'].avg,
        **c_indices,
    }

    # Add loss components and metrics
    for key in meters.keys():
        if key != 'bag_size':
            results_dict[key] = meters[key].avg

    if verbose:
        print(f'\nValidation Results:')
        print(f'  Total Loss: {results_dict["loss"]:.4f}')
        print(f'  C-Index (Causal): {c_indices["c_index_causal"]:.4f}')
        print(f'  C-Index (Confound): {c_indices["c_index_confound"]:.4f}')
        print(f'  C-Index (Full): {c_indices["c_index_full"]:.4f}')
        print(f'  Avg Histo Causal Prototypes: {meters["histo_causal_count"].avg:.2f}')
        print(f'  Avg Gene Causal Prototypes: {meters["gene_causal_count"].avg:.2f}')

    return results_dict, dumps if dump_results else None


if __name__ == '__main__':
    """
    Test the trainer functions
    """
    print("Causal Separation Trainer module loaded successfully!")

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
from utils.losses import NLLSurvLoss, CoxLoss, SurvRankingLoss
from utils.utils import (EarlyStopping, save_checkpoint, AverageMeter, safe_list_to,
                         get_optim, print_network, get_lr_scheduler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROTO_MODELS = ['PANTHER', 'OT', 'H2T', 'ProtoCount']

def train(datasets, args):
    """
    Train for a single fold for suvival
    """
    
    writer_dir = args.results_dir
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    assert args.es_metric == 'loss'
    
    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()
    elif args.loss_fn == 'rank':
        loss_fn = SurvRankingLoss()

    args.feat_dim = args.in_dim # Patch feature dimension
    print('\nInit Model...', end=' ')

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
            loader.dataset.X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean), torch.Tensor(cov)], dim=-1)

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
        train_results = train_loop_survival(model, datasets['train'], optimizer, lr_scheduler, loss_fn,
                                            print_every=args.print_every, accum_steps=args.accum_steps)

        writer = log_dict_tensorboard(writer, train_results, 'train/', epoch)

        ### Validation Loop (Optional)
        if 'val' in datasets.keys():
            print('#' * 11, f'VAL Epoch: {epoch}', '#' * 11)
            val_results, _ = validate_survival(model, datasets['val'], loss_fn,
                                                   print_every=args.print_every, verbose=True)

            writer = log_dict_tensorboard(writer, val_results, 'val/', epoch)

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

    ### End of epoch: Load in the best model (or save the latest model with not early stopping)
    if args.early_stopping:
        model.load_state_dict(torch.load(j_(args.results_dir, f"s_checkpoint.pth"))['model'])
    else:
        torch.save(model.state_dict(), j_(args.results_dir, f"s_checkpoint.pth"))

    ### End of epoch: Evaluate on val and test set
    results, dumps = {}, {}
    for k, loader in datasets.items():
        print(f'End of training. Evaluating on Split {k.upper()}...:')
        return_attn = True # True for MMP
        results[k], dumps[k] = validate_survival(model, loader, loss_fn, print_every=args.print_every,
                                                     dump_results=True, return_attn=return_attn, verbose=False)

        if k == 'train':
            _ = results.pop('train')  # Train results by default are not saved in the summary, but train dumps are
        
    writer.close()
    return results, dumps

## SURVIVAL
def train_loop_survival(model, loader, optimizer, lr_scheduler, loss_fn=None, 
                        print_every=50, accum_steps=32):
    
    model.train()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    
    for batch_idx, batch in enumerate(loader):
        data = safe_list_to(batch['img'], device)
        label = safe_list_to(batch['label'], device)

        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)
        attn_mask = batch['attn_mask'].to(device) if ('attn_mask' in batch) else None

        omics = safe_list_to(batch['omics'], device)

        out, log_dict = model(data, omics, attn_mask=attn_mask, label=label, censorship=censorship, loss_fn=loss_fn)

        if out['loss'] is None:
            continue

        # Get loss + backprop
        loss = out['loss']
        loss = loss / accum_steps
        loss.backward()
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # End of iteration survival-specific metrics to calculate / log
        all_risk_scores.append(out['risk'].detach().cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())

        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))

        bag_size_meter.update(data.size(1), n=len(data))

        if ((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})
    results['lr'] = optimizer.param_groups[0]['lr']
    return results


@torch.no_grad()
def validate_survival(model, loader,
                      loss_fn=None,
                      print_every=50,
                      dump_results=False,
                      recompute_loss_at_end=True,
                      return_attn=False,
                      verbose=1):
    model.eval()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    all_omic_attn, all_cross_attn, all_path_attn = [], [], []

    for batch_idx, batch in enumerate(loader):
        data = batch['img'].to(device)
        label = batch['label'].to(device)
        omics = safe_list_to(batch['omics'], device)

        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)
        attn_mask = batch['attn_mask'].to(device) if ('attn_mask' in batch) else None
        
        out, log_dict = model(data, omics, attn_mask=attn_mask, label=label, censorship=censorship, loss_fn=loss_fn, return_attn=return_attn)
        if return_attn:
            all_omic_attn.append(out['omic_attn'].detach().cpu().numpy())
            all_cross_attn.append(out['cross_attn'].detach().cpu().numpy())
            all_path_attn.append(out['path_attn'].detach().cpu().numpy())
        # End of iteration survival-specific metrics to calculate / log
        bag_size_meter.update(data.size(1), n=len(data))
        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))
        all_risk_scores.append(out['risk'].cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())

        if verbose and (((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1)):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    if return_attn:
        if len(all_omic_attn[0].shape) == 2:
            all_omic_attn = np.stack(all_omic_attn)
            all_cross_attn = np.stack(all_cross_attn)
            all_path_attn = np.stack(all_path_attn)
        else:
            all_omic_attn = np.vstack(all_omic_attn)
            all_cross_attn = np.vstack(all_cross_attn)
            all_path_attn = np.vstack(all_path_attn)

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})

    if recompute_loss_at_end and isinstance(loss_fn, CoxLoss):
        surv_loss_dict = loss_fn(logits=torch.tensor(all_risk_scores).unsqueeze(1),
                                 times=torch.tensor(all_event_times).unsqueeze(1),
                                 censorships=torch.tensor(all_censorships).unsqueeze(1))
        results['surv_loss'] = surv_loss_dict['loss'].item()
        results.update({k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)})

    if verbose:
        msg = [f"{k}: {v:.3f}" for k, v in results.items()]
        print("\t".join(msg))

    dumps = {}
    if dump_results:
        dumps['all_risk_scores'] = all_risk_scores
        dumps['all_censorships'] = all_censorships
        dumps['all_event_times'] = all_event_times
        dumps['sample_ids'] = np.array(
            loader.dataset.idx2sample_df['sample_id'])
        if return_attn:
            dumps['all_omic_attn'] = all_omic_attn
            dumps['all_cross_attn'] = all_cross_attn
            dumps['all_path_attn'] = all_path_attn
    return results, dumps
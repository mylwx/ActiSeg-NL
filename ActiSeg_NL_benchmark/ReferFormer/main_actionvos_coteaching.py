import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import  train_one_epoch_coteaching
from engine import evaluate, evaluate_a2d
from models import build_model
from tools.load_pretrained_weights import pre_trained_model_to_finetune
import opts


def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def main(args):
    args.masks = True
    # args.batch_size = 2
    utils.init_distributed_mode(args)
    print("args = ", args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(f'\n Run on {args.dataset_file} dataset.')
    print('\n')

    device = torch.device(args.device)

    # Set different seeds for reproducibility of model1 and model2.
    seed1 = args.seed + utils.get_rank()
    seed2 = args.seed + utils.get_rank() + 1  # Different seed for model2.

    set_seed(seed1)
    model1, criterion1, postprocessor1 = build_model(args)
    model1.to(device)

    set_seed(seed2)
    model2, criterion2, postprocessor2 = build_model(args)
    model2.to(device)

    # Construct parameter dictionaries using the same logic as in the original main function.
    def construct_param_dicts(model_without_ddp, args):
        param_dicts = [
            {
                "params":
                    [p for n, p in model_without_ddp.named_parameters()
                     if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_text_encoder_names)
                     and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                "lr": args.lr_backbone,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_text_encoder_names) and p.requires_grad],
                "lr": args.lr_text_encoder,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr * args.lr_linear_proj_mult,
            }
        ]
        return param_dicts

    model_without_ddp1 = model1
    model_without_ddp2 = model2

    if args.distributed:
        model1 = torch.nn.parallel.DistributedDataParallel(model1, device_ids=[args.gpu])
        model2 = torch.nn.parallel.DistributedDataParallel(model2, device_ids=[args.gpu])
        model_without_ddp1 = model1.module
        model_without_ddp2 = model2.module

    param_dicts1 = construct_param_dicts(model_without_ddp1, args)
    param_dicts2 = construct_param_dicts(model_without_ddp2, args)

    optimizer1 = torch.optim.AdamW(param_dicts1, lr=args.lr, weight_decay=args.weight_decay)
    optimizer2 = torch.optim.AdamW(param_dicts2, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, args.lr_drop)
    lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, args.lr_drop)

    # Build actionvos train-val
    dataset_train = build_dataset(args.dataset_file, image_set='train', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # Load pretrained weights for both models if necessary.
    for model, model_without_ddp in [(model1, model_without_ddp1), (model2, model_without_ddp2)]:
        if args.dataset_file != "davis" and args.dataset_file != "jhmdb" and args.pretrained_weights is not None:
            checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
            checkpoint_dict = pre_trained_model_to_finetune(checkpoint, args)
            model_without_ddp.load_state_dict(checkpoint_dict, strict=False)

    output_dir = Path(args.output_dir)

    if args.resume1:###############add 
        if args.resume1.startswith('https'):
            checkpoint1 = torch.hub.load_state_dict_from_url(
                args.resume1, map_location='cpu', check_hash=True)
        else:
            checkpoint1 = torch.load(args.resume1, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp1.load_state_dict(checkpoint1['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint1 and 'lr_scheduler' in checkpoint1 and 'epoch' in checkpoint1:
            import copy
            p_groups = copy.deepcopy(optimizer1.param_groups)
            optimizer1.load_state_dict(checkpoint1['optimizer'])
            for pg, pg_old in zip(optimizer1.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer1.param_groups)
            lr_scheduler1.load_state_dict(checkpoint1['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler1.step_size = args.lr_drop
                lr_scheduler1.base_lrs = list(map(lambda group: group['initial_lr'], optimizer1.param_groups))
            lr_scheduler1.step(lr_scheduler1.last_epoch)
            args.start_epoch = checkpoint1['epoch'] + 1
    if args.resume2:###############add 
        if args.resume2.startswith('https'):
            checkpoint2 = torch.hub.load_state_dict_from_url(
                args.resume2, map_location='cpu', check_hash=True)
        else:
            checkpoint2 = torch.load(args.resume2, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp2.load_state_dict(checkpoint2['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint2 and 'lr_scheduler' in checkpoint2 and 'epoch' in checkpoint2:
            import copy
            p_groups = copy.deepcopy(optimizer2.param_groups)
            optimizer1.load_state_dict(checkpoint2['optimizer'])
            for pg, pg_old in zip(optimizer2.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer2.param_groups)
            lr_scheduler2.load_state_dict(checkpoint2['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler2.step_size = args.lr_drop
                lr_scheduler2.base_lrs = list(map(lambda group: group['initial_lr'], optimizer1.param_groups))
            lr_scheduler2.step(lr_scheduler2.last_epoch)
            args.start_epoch = checkpoint2['epoch'] + 1

    if args.eval:
        assert args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb', \
               'Only A2D-Sentences and JHMDB-Sentences datasets support evaluation'
        test_stats1 = evaluate_a2d(model1, data_loader_val, postprocessor1, device, args)
        test_stats2 = evaluate_a2d(model2, data_loader_val, postprocessor2, device, args)
        return

    Tk = 6
    threshold = 0.95# noisy rate
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch+1, args.epochs+1):########Start from 1

        # Rt =1 - min(threshold*epoch/Tk,threshold)
        if epoch <= Tk:
            Rt = threshold-0.1*(epoch-1)/Tk
        else:    
            Rt = threshold-0.1
        print(f"Rt={Rt}")
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch_coteaching(
            model1, model2, criterion1, data_loader_train, optimizer1, optimizer2, device, epoch,
            args.clip_max_norm,Rt
        )
        lr_scheduler1.step()
        lr_scheduler2.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_model1.pth']
            if (epoch + 1) % args.save_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint_model1_{(epoch+1):04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp1.state_dict(),
                    'optimizer': optimizer1.state_dict(),
                    'lr_scheduler': lr_scheduler1.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

            checkpoint_paths = [output_dir / 'checkpoint_model2.pth']
            if (epoch + 1) % args.save_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint_model2_{(epoch+1):04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp2.state_dict(),
                    'optimizer': optimizer2.state_dict(),
                    'lr_scheduler': lr_scheduler2.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if args.dataset_file == 'a2d':
            test_stats1 = evaluate_a2d(model1, data_loader_val, postprocessor1, device, args)
            test_stats2 = evaluate_a2d(model2, data_loader_val, postprocessor2, device, args)
            log_stats.update({**{f'{k}_model1': v for k, v in test_stats1.items()},
                              **{f'{k}_model2': v for k, v in test_stats2.items()}})

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ReferFormer training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    args.dataset_file = 'actionvos'
    if args.all_pos:
        args.dataset_file = 'actionvos_allpos'
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)




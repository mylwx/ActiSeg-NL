# opts.py
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('ReferFormer training and inference scripts.', add_help=False)

    # Learning rate / optimizer
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=5e-5, type=float)
    parser.add_argument('--lr_backbone_names', default=['backbone.0'], type=str, nargs='+')
    parser.add_argument('--lr_text_encoder', default=1e-5, type=float)
    parser.add_argument('--lr_text_encoder_names', default=['text_encoder'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=1.0, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--lr_drop', default=[3], type=int, nargs='+')

    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd'])
    parser.add_argument('--momentum', default=0.9, type=float)

    parser.add_argument('--scheduler_type', default='multistep', type=str,
                        choices=['multistep', 'cosine_annealing', 'cyclic'])
    parser.add_argument('--t_0', default=5, type=int)
    parser.add_argument('--t_mult', default=1, type=float)
    parser.add_argument('--eta_min', default=1e-6, type=float)
    parser.add_argument('--step_size_up_ratio', default=0.4, type=float)

    # Training common
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--save_interval', default=5, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    # Text encoding model (offline)
    parser.add_argument('--text_model_name', default='roberta-base', type=str,
                        choices=['roberta-base', 'snowflake-arctic-embed-l', 'bge-m3'])

    # New structure switches (key! default all False = original version)
    parser.add_argument('--use_object_query_from_words', action='store_true',
                        help='use word-level aggregation (or TNS) to build text_embed; otherwise use sentence embedding (original)')
    parser.add_argument('--use_tns', action='store_true',
                        help='enable TNS when use_object_query_from_words=True')
    parser.add_argument('--use_saa', action='store_true',
                        help='enable SAA for text cleaning in fusion/pixel decoder')

    # Redundant tokens / contrastive loss
    parser.add_argument('--num_redundant_tokens', default=3, type=int)
    parser.add_argument('--use_redundant_tokens', action='store_true')
    parser.add_argument('--use_contrast_loss', action='store_true')
    parser.add_argument('--contrast_loss_coef', default=1.0, type=float)

    # Region prototypes / TAF / export features for loss side (FAR/CRM)
    parser.add_argument('--use_region_prototypes', action='store_true',
                        help='export region-level prototypes (fg/bg) for FAR/CRM/diagnostics')
    parser.add_argument('--use_taf', action='store_true',
                        help='enable Temporal Agreement Factor when aggregating region prototypes')
    parser.add_argument('--taf_eta', default=0.2, type=float,
                        help='temperature for TAF: softmax(eta * log(conf_t)) over frames')
    parser.add_argument('--export_mask_features_for_loss', action='store_true',
                        help='export mask_features (detached) to outputs for FAR/CRM losses')

    # Router (CE↔GCE routing)
    parser.add_argument('--use_loss_router', action='store_true',
                        help='enable router statistics and (optionally) loss reweighting')
    parser.add_argument('--router_a', default=2.0, type=float)
    parser.add_argument('--router_b', default=2.0, type=float)
    parser.add_argument('--router_c', default=1.0, type=float)
    parser.add_argument('--router_d', default=0.0, type=float)
    parser.add_argument('--router_conf_alpha', default=1.0, type=float)
    parser.add_argument('--gce_q', default=0.7, type=float,
                        help='q in GCE; q→0 recovers CE')

    # === NLPrompt (CE↔MAE + batch-wise OT)
    parser.add_argument('--use_nlprompt', action='store_true',
                        help='enable NLPrompt-style CE/MAE routing with batch-wise OT purification')
    parser.add_argument('--nlp_ot_reg', default=0.1, type=float,
                        help='entropy regularization (epsilon) for Sinkhorn')
    parser.add_argument('--nlp_ot_iters', default=20, type=int,
                        help='Sinkhorn iterations')
    parser.add_argument('--nlp_clean_tau', default=0.5, type=float,
                        help='temperature/sharpening for converting OT diag to clean weight')
    # Pairwise switch: default True, can be disabled with --no_nlp_fuse_router
    parser.add_argument('--nlp_fuse_router', dest='nlp_fuse_router', action='store_true',
                        help='fuse NLPrompt clean weight with router clean weight by element-wise product')
    parser.add_argument('--no_nlp_fuse_router', dest='nlp_fuse_router', action='store_false',
                        help='do NOT fuse NLPrompt clean weight with router clean weight')
    parser.set_defaults(nlp_fuse_router=True)
    parser.add_argument('--nlp_mae_lambda', default=1.0, type=float,
                        help='relative weight of MAE term vs CE in NLPrompt routing')

    # === CTRL (Clustering-based Trajectory Regularization for label noise)
    parser.add_argument('--use_ctrl', action='store_true',
                        help='enable CTRL-style loss trajectory bank and clustering guidance')
    parser.add_argument('--ctrl_fusion', default='geom', type=str,
                        choices=['geom', 'cos', 'none'],
                        help="how to fuse CTRL confidence with router/NLPrompt; 'geom'=geometric mean, 'cos'=cosine rescale, 'none'=no fuse")
    parser.add_argument('--ctrl_strength', default=0.5, type=float,
                        help='CTRL influence strength (0-1)')
    parser.add_argument('--ctrl_bank_momentum', default=0.9, type=float,
                        help='EMA momentum for per-sample loss trajectory bank')
    parser.add_argument('--ctrl_bank_path', default=None, type=str,
                        help='path to save/load CTRL trajectory bank (e.g., runs/xxx/ctrl_bank.pkl)')

    # BUL / FAR / CRM
    parser.add_argument('--use_bul', action='store_true', help='enable BUL soft targets for masks')
    parser.add_argument('--bul_kernel_size', default=5, type=int)
    parser.add_argument('--bul_sigma', default=1.0, type=float)

    parser.add_argument('--use_far', action='store_true', help='enable Feature Affinity Regularization')
    parser.add_argument('--far_tau', default=0.1, type=float)
    parser.add_argument('--far_weight', default=0.2, type=float)

    parser.add_argument('--use_crm', action='store_true', help='enable Confident-Region Mining (prototype contrast)')
    parser.add_argument('--crm_q_fg', default=0.8, type=float)
    parser.add_argument('--crm_q_bg', default=0.2, type=float)
    parser.add_argument('--crm_tau', default=0.07, type=float)
    parser.add_argument('--crm_weight', default=0.2, type=float)

    # Data path supplements
    parser.add_argument('--imagesets_path', type=str, default=None)
    parser.add_argument('--jpegimages_path', type=str, default=None)
    parser.add_argument('--annotations_path', type=str, default=None)
    parser.add_argument('--weights_path', type=str, default=None)

    # AMP
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--loss_gather_dir', type=str, default=None)
    parser.add_argument('--pretrained_weights', type=str, default=None)

    # Variants
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--backbone_pretrained', default=None, type=str)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--num_feature_levels', default=4, type=int)

    # Transformer
    parser.add_argument('--enc_layers', default=4, type=int)
    parser.add_argument('--enc_layers', default=4, type=int)
    parser.add_argument('--dec_layers', default=4, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_frames', default=5, type=int)
    parser.add_argument('--num_queries', default=5, type=int)
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--freeze_text_encoder', action='store_true')

    # Segmentation & losses (keep original)
    parser.add_argument('--use_weights', action='store_true')
    parser.add_argument('--use_weights', action='store_true')
    parser.add_argument('--use_gce', action='store_true')
    parser.add_argument('--use_gce_weights', action='store_true')
    parser.add_argument('--use_sce', action='store_true')
    parser.add_argument('--use_sce_weights', action='store_true')
    parser.add_argument('--use_elr_loss', action='store_true')
    parser.add_argument('--use_elr_loss_mask', action='store_true')
    parser.add_argument('--use_npn_loss', action='store_true')
    parser.add_argument('--use_npn_loss_mask', action='store_true')
    parser.add_argument('--use_active_passive', action='store_true')

    parser.add_argument('--all_pos', action='store_true')
    parser.add_argument('--use_positive_cls', action='store_true')
    parser.add_argument('--pos_cls_thres', default=0.75, type=float)
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--mask_dim', default=256, type=int)
    parser.add_argument('--controller_layers', default=3, type=int)
    parser.add_argument('--dynamic_mask_channels', default=8, type=int)
    parser.add_argument('--no_rel_coord', dest='rel_coord', action='store_false')
    parser.add_argument('--has_class_category_embed', default=False, action='store_true')
    parser.add_argument('--num_category_classes', default=1, type=int)

    # Loss coefficients
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--set_cost_mask', default=2, type=float)
    parser.add_argument('--set_cost_dice', default=5, type=float)
    parser.add_argument('--mask_loss_coef', default=2, type=float)
    parser.add_argument('--dice_loss_coef', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--cls_category_loss_coef', default=2, type=float)
    parser.add_argument('--cls_gce_loss_coef', default=2, type=float)
    parser.add_argument('--cls_sce_loss_coef', default=2, type=float)
    parser.add_argument('--cls_active_passive_loss_coef', default=2, type=float)

    # Dataset
    parser.add_argument('--dataset_file', default='ytvos', help='Dataset name')
    parser.add_argument('--dataset_file', default='ytvos', help='Dataset name')
    parser.add_argument('--expression_file', default='meta_expressions.json')
    parser.add_argument('--actionvos_path', type=str, default='../dataset_visor')
    parser.add_argument('--coco_path', type=str, default='data/coco')
    parser.add_argument('--ytvos_path', type=str, default='data/ref-youtube-vos')
    parser.add_argument('--davis_path', type=str, default='data/ref-davis')
    parser.add_argument('--a2d_path', type=str, default='data/a2d_sentences')
    parser.add_argument('--jhmdb_path', type=str, default='data/jhmdb_sentences')
    parser.add_argument('--max_skip', default=3, type=int)
    parser.add_argument('--max_size', default=512, type=int)
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--resume1', default='')
    parser.add_argument('--resume2', default='')
    parser.add_argument('--index_map_path', default='')
    parser.add_argument('--sample_map_path', default='')
    parser.add_argument('--base_height', default=128, type=int)
    parser.add_argument('--base_width', default=128, type=int)
    parser.add_argument('--dataset_ws', action='store_true')
    parser.add_argument('--warmup_epochs_npn', default=2, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=16, type=int)

    # Testing
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--ngpu', default=8, type=int)
    parser.add_argument('--split', default='val', type=str, choices=['val', 'test'])
    parser.add_argument('--visualize', action='store_true')

    # Distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--cache_mode', default=False, action='store_true')
    # DDP parameters
    parser.add_argument('--ddp_find_unused_parameters', action='store_true',
    parser.add_argument('--ddp_find_unused_parameters', action='store_true',
                        help='set DistributedDataParallel(find_unused_parameters=True) to avoid reduction errors')

    return parser

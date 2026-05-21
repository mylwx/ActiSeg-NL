import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('ReferFormer training and inference scripts.', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=5e-5, type=float)
    parser.add_argument('--lr_backbone_names', default=['backbone.0'], type=str, nargs='+')
    parser.add_argument('--lr_text_encoder', default=1e-5, type=float)
    parser.add_argument('--lr_text_encoder_names', default=['text_encoder'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=1.0, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--save_interval', default=5, type=int)
    parser.add_argument('--lr_drop', default=3, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Dataset parameters (newly added path parameters)
    parser.add_argument('--imagesets_path', type=str, default=None, help='Path to ImageSets directory')
    parser.add_argument('--jpegimages_path', type=str, default=None, help='Path to JPEGImages_Sparse directory')
    parser.add_argument('--annotations_path', type=str, default=None, help='Path to Annotations_Sparse directory')
    parser.add_argument('--weights_path', type=str, default=None, help='Path to Weights_Sparse directory')
    parser.add_argument('--clean_annotations_path', type=str, default=None, help='Path to clean Annotations_Sparse directory')

    # Model parameters
    # load the pretrained weights
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help="Path to the pretrained model.")

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true') # NOTE: must be false

    # * Backbone
    # ["resnet50", "resnet101", "swin_t_p4w7", "swin_s_p4w7", "swin_b_p4w7", "swin_l_p4w7"]
    # ["video_swin_t_p4w7", "video_swin_s_p4w7", "video_swin_b_p4w7"]
    parser.add_argument('--backbone', default='resnet50', type=str, 
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--backbone_pretrained', default=None, type=str, 
                        help="if use swin backbone and train from scratch, the path to the pretrained weights")
    parser.add_argument('--use_checkpoint', action='store_true', help='whether use checkpoint for swin/video swin backbone')
    parser.add_argument('--dilation', action='store_true', # DC5
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, 
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=5, type=int,
                        help="Number of clip frames for training")
    parser.add_argument('--num_queries', default=5, type=int,
                        help="Number of query slots, all frames share the same queries") 
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    # for text
    parser.add_argument('--freeze_text_encoder', action='store_true') # default: False

    # * Segmentation
    parser.add_argument('--use_weights', action='store_true', help="use action-guided weights for focal loss")
    parser.add_argument('--all_pos', action='store_true',
                        help="in training, keep all masks as positive")
    parser.add_argument('--use_positive_cls', action='store_true',
                        help="use an extra positive classification head")
    parser.add_argument('--pos_cls_thres', default=0.75, type=float, 
                        help="in inference, use positive classification results and the classification threshold")
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--mask_dim', default=256, type=int, 
                        help="Size of the mask embeddings (dimension of the dynamic mask conv)")
    parser.add_argument('--controller_layers', default=3, type=int, 
                        help="Dynamic conv layer number")
    parser.add_argument('--dynamic_mask_channels', default=8, type=int, 
                        help="Dynamic conv final channel number")
    parser.add_argument('--no_rel_coord', dest='rel_coord', action='store_false',
                        help="Disables relative coordinates")
    
    ## Append PMHM related switches and coefficients in parameter list
    #######################################
    # ===== PMHM and robust training =====
    parser.add_argument('--use_pmhm', type=int, default=1)
    parser.add_argument('--n_aux_heads', type=int, default=1)                # 0 or 1 or 2
    parser.add_argument('--init_mode_single', type=str, default='copy_perturb')  # 'random' or 'copy_perturb'
    parser.add_argument('--copy_sigma', type=float, default=1e-3)
    parser.add_argument('--aux_p_drop', type=float, default=0.2)
    parser.add_argument('--aux_gamma_noise', type=float, default=0.2)
    parser.add_argument('--use_proto', type=int, default=0)

    parser.add_argument('--k_agree', type=int, default=2)
    parser.add_argument('--tau_m_p', type=float, default=0.20)
    parser.add_argument('--tau_e_p', type=float, default=0.85)
    parser.add_argument('--tau_h_p', type=float, default=0.60)
    parser.add_argument('--tv_loss_coef', type=float, default=0.3)
    parser.add_argument('--head_loss_coef', type=float, default=0.1)
    parser.add_argument('--layer_loss_coef', type=float, default=0.1)
    parser.add_argument('--tv_alpha_start', type=float, default=0.7)
    parser.add_argument('--tv_alpha_end', type=float, default=0.5)
    parser.add_argument('--tv_beta', type=float, default=0.5)
    parser.add_argument('--proto_loss_coef', type=float, default=0.05)
    ## Use parameter switches to control three modes: none, epoch, step. Default is epoch, you can switch in command
    # ===== PMHM dynamic freezing switches and hyperparameters =====
    parser.add_argument('--aux_freeze_mode', type=str, default='epoch', choices=['none', 'epoch', 'step'])
    parser.add_argument('--aux_freeze_p0', type=float, default=0.3)   # epoch mode start freezing probability
    parser.add_argument('--aux_freeze_p1', type=float, default=0.1)   # epoch mode end freezing probability
    parser.add_argument('--aux_freeze_step_p', type=float, default=0.2)  # step mode per-step freezing probability

    
    #######################################

    ## PMHM key variable statistics & visualization
    #######################################
    # ---- PMHM debug / viz ----
    parser.add_argument('--pmhm_log_stats', action='store_true',
                        help='log PMHM scalar stats (S/U ratio, taus, vote, KL, etc.)')
    parser.add_argument('--pmhm_viz_images', action='store_true',
                        help='save PMHM debug images (P, M, E, S, U, votes)')
    parser.add_argument('--pmhm_viz_every', type=int, default=2000,
                        help='save one debug image every N calls to masks_robust')
    parser.add_argument('--pmhm_viz_max', type=int, default=20,
                        help='cap number of debug images per epoch')
    parser.add_argument('--pmhm_viz_dir', type=str, default='pmhm_viz',
                        help='subdir under output_dir to save PMHM images')

    #######################################


    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_mask', default=2, type=float,
                        help="mask coefficient in the matching cost")
    parser.add_argument('--set_cost_dice', default=5, type=float,
                        help="mask coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=2, type=float)
    parser.add_argument('--dice_loss_coef', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    # ['ytvos', 'davis', 'a2d', 'jhmdb', 'refcoco', 'refcoco+', 'refcocog', 'all']
    # 'all': using the three ref datasets for pretraining
    parser.add_argument('--dataset_file', default='ytvos', help='Dataset name')
    parser.add_argument('--expression_file', default='meta_expressions.json', help='Annotation exp name')  
    parser.add_argument('--actionvos_path', type=str, default='../dataset_visor')
    parser.add_argument('--coco_path', type=str, default='data/coco')
    parser.add_argument('--ytvos_path', type=str, default='data/ref-youtube-vos')
    parser.add_argument('--davis_path', type=str, default='data/ref-davis')
    parser.add_argument('--a2d_path', type=str, default='data/a2d_sentences')
    parser.add_argument('--jhmdb_path', type=str, default='data/jhmdb_sentences')
    parser.add_argument('--max_skip', default=3, type=int, help="max skip frame number")
    parser.add_argument('--max_size', default=512, type=int, help="max size for the frame")
    # changed max_size from 640 to 512 due to cuda OOM with video-swin-b backbone
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=16, type=int)

    # test setting
    parser.add_argument('--threshold', default=0.5, type=float) # binary threshold for mask
    parser.add_argument('--ngpu', default=8, type=int, help='gpu number when inference for ref-ytvos and ref-davis')
    parser.add_argument('--split', default='val', type=str, choices=['val', 'test'])
    parser.add_argument('--visualize', action='store_true', help='whether visualize the masks during inference')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    return parser



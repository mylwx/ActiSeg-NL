"""
ActionVOS data loader with fixed sampling for ELR
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset
import datasets.transforms_video_actionvos as T
import os
from PIL import Image
import json
import numpy as np
import random

class ActionVOSDatasetClipID(Dataset):
    def __init__(self, actionvos_folder: Path, ann_file: Path, transforms, return_masks: bool, 
                 num_frames: int, max_skip: int, use_weights: bool, image_set: str,
                 index_map_path="clip_index_map.json", sample_map_path="clip_sample_map.json"):
        self.actionvos_folder = actionvos_folder     
        self.ann_file = ann_file         
        self._transforms = transforms    
        self.return_masks = return_masks
        self.num_frames = num_frames     
        self.max_skip = max_skip
        self.use_weights = use_weights
        self.image_set = image_set
        
        # Load or generate index mapping table
        self.index_map_path = index_map_path
        self.sample_map_path = sample_map_path
        self._load_or_create_index_map()
        self._load_or_create_sample_map()

        # Create video metadata
        self.prepare_metas()       

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))  
        print('\n')    

    def _load_or_create_index_map(self):
        """Load or generate video clip global index mapping table"""
        if os.path.exists(self.index_map_path):
            with open(self.index_map_path, "r") as f:
                self.index_map = json.load(f)
            print(f"Loaded index map from {self.index_map_path} with {len(self.index_map)} clips")
        else:
            self.index_map = {}
            with open(os.path.join(str(self.actionvos_folder), 'ImageSets', f'{self.image_set}_objects_category.json'), 'r') as f:
                subset_metas_by_video = json.load(f)['videos']
            with open(os.path.join(str(self.actionvos_folder), 'ImageSets', self.ann_file), 'r') as f:
                subset_expressions_by_video = json.load(f)['videos']
            videos = list(subset_expressions_by_video.keys())
            
            clip_idx = 0
            for vid in videos:
                vid_data = subset_expressions_by_video[vid]
                vid_frames = sorted(vid_data['frames'])
                vid_len = len(vid_frames)
                for exp_id, exp_dict in vid_data['expressions'].items():
                    for frame_id in range(0, vid_len, self.num_frames):
                        clip_key = f"{vid}_{exp_id}_{frame_id}"
                        self.index_map[clip_key] = clip_idx
                        clip_idx += 1
            
            with open(self.index_map_path, "w") as f:
                json.dump(self.index_map, f)
            print(f"Created and saved index map to {self.index_map_path} with {len(self.index_map)} clips")

    def _load_or_create_sample_map(self):
        """Load or generate fixed frame sampling mapping table"""
        if os.path.exists(self.sample_map_path):
            with open(self.sample_map_path, "r") as f:
                self.sample_map = json.load(f)
            print(f"Loaded sample map from {self.sample_map_path} with {len(self.sample_map)} clips")
        else:
            self.sample_map = {}
            with open(os.path.join(str(self.actionvos_folder), 'ImageSets', self.ann_file), 'r') as f:
                subset_expressions_by_video = json.load(f)['videos']
            videos = list(subset_expressions_by_video.keys())
            random.seed(42)  # Fixed random seed

            for vid in videos:
                vid_data = subset_expressions_by_video[vid]
                vid_frames = sorted(vid_data['frames'])
                vid_len = len(vid_frames)
                for exp_id, exp_dict in vid_data['expressions'].items():
                    for frame_id in range(0, vid_len, self.num_frames):
                        clip_key = f"{vid}_{exp_id}_{frame_id}"
                        sample_indx = [frame_id]
                        if self.num_frames != 1:
                            sample_id_before = random.randint(1, 3)
                            sample_id_after = random.randint(1, 3)
                            local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                            sample_indx.extend(local_indx)
                            if self.num_frames > 3:
                                all_inds = list(range(vid_len))
                                global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                                global_n = self.num_frames - len(sample_indx)
                                if len(global_inds) > global_n:
                                    select_id = random.sample(range(len(global_inds)), global_n)
                                    for s_id in select_id:
                                        sample_indx.append(global_inds[s_id])
                                elif vid_len >= global_n:
                                    select_id = random.sample(range(vid_len), global_n)
                                    for s_id in select_id:
                                        sample_indx.append(all_inds[s_id])
                                else:
                                    select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
                                    for s_id in select_id:
                                        sample_indx.append(all_inds[s_id])
                        sample_indx.sort()
                        self.sample_map[clip_key] = sample_indx

            with open(self.sample_map_path, "w") as f:
                json.dump(self.sample_map, f)
            print(f"Created and saved sample map to {self.sample_map_path} with {len(self.sample_map)} clips")

    def prepare_metas(self):
        with open(os.path.join(str(self.actionvos_folder), 'ImageSets', f'{self.image_set}_objects_category.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        
        with open(os.path.join(str(self.actionvos_folder), 'ImageSets', self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['exp_id'] = exp_id  # Add exp_id to meta
                    meta['obj_id'] = int(exp_dict['obj_id'])
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    meta['clip_idx'] = self.index_map[f"{vid}_{exp_id}_{frame_id}"]
                    obj_id = exp_dict['obj_id']
                    meta['category'] = vid_meta['objects'][obj_id]['category']
                    meta['class_id'] = exp_dict['class_id']
                    meta['positive'] = exp_dict['positive']
                    # meta['class_id_org'] = exp_dict['class_id_org']
                    self.metas.append(meta)

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax  # y1, y2, x1, x2 
        
    def __len__(self):
        return len(self.metas)
        
    def __getitem__(self, idx):
        meta = self.metas[idx]
        video, exp, exp_id, obj_id, category, positive, frames, frame_id, clip_idx = \
                    meta['video'], meta['exp'], meta['exp_id'], meta['obj_id'], meta['category'], meta['positive'], meta['frames'], meta['frame_id'], meta['clip_idx']
        exp = " ".join(exp.lower().split())
        category_id = meta['class_id']
        # category_id_org = meta['class_id_org']
        vid_len = len(frames)

        # Use fixed frame indices
        clip_key = f"{video}_{exp_id}_{frame_id}"
        sample_indx = self.sample_map[clip_key]

        imgs, labels, boxes, masks, valid = [], [], [], [], []
        labels_org, is_name_noises = [], []
        positives = []
        weights = []
        for j in range(self.num_frames):
            frame_indx = sample_indx[j]
            frame_name = frames[frame_indx]
            img_path = os.path.join(str(self.actionvos_folder), 'JPEGImages_Sparse', self.image_set, video, frame_name + '.jpg')
            mask_path = os.path.join(str(self.actionvos_folder), 'Annotations_Sparse', self.image_set, video, frame_name + '.png')
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('P')
            if self.use_weights:
                weight_path = os.path.join(str(self.actionvos_folder), 'Weights_Sparse', self.image_set, video, frame_name + '.png')
                weight = Image.open(weight_path).convert('P')
                weight = np.array(weight)

            label = torch.tensor(category_id)
            # label_org = torch.tensor(category_id_org)
            # if category_id == category_id_org:
            #     is_name_noise = torch.tensor(0)
            # else:
            #     is_name_noise = torch.tensor(1)
            mask = np.array(mask)
            
            if self.use_weights:
                mask = (mask == obj_id).astype(np.float32)
                weight = np.where(mask == 0, 1, weight)
                mask = np.where(weight == 3, 0, mask)
                weight = np.where(weight >= 3, 5, weight)
                if positive:
                    if (mask > 0).any():
                        y1, y2, x1, x2 = self.bounding_box(mask)
                        box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    else:
                        box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                else:
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                    mask = np.zeros_like(mask)
            else:
                if positive:
                    mask = (mask == obj_id).astype(np.float32)
                    if (mask > 0).any():
                        y1, y2, x1, x2 = self.bounding_box(mask)
                        box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    else:
                        box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                else:
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                    mask = np.zeros_like(mask)
                weight = np.ones_like(mask)
            mask = torch.from_numpy(mask)
            weight = torch.from_numpy(weight)

            imgs.append(img)
            labels.append(label)
            masks.append(mask)
            boxes.append(box)
            valid.append(1)
            positives.append(positive)
            weights.append(weight)
            # labels_org.append(label_org)
            # is_name_noises.append(is_name_noise)

        w, h = img.size
        labels = torch.stack(labels, dim=0)
        boxes = torch.stack(boxes, dim=0)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        masks = torch.stack(masks, dim=0)
        weights = torch.stack(weights, dim=0)
        # labels_org = torch.stack(labels_org, dim=0)
        # is_name_noises = torch.stack(is_name_noises, dim=0)

        target = {
            'frames_idx': torch.tensor(sample_indx),
            'labels': labels,
            'boxes': boxes,
            'masks': masks,
            'weights': weights,
            'valid': torch.tensor(valid),
            'positive': torch.tensor(positives),
            'caption': exp,
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)]),
            # 'labels_org': labels_org,
            # 'is_name_noises': is_name_noises,
            'video_id': video,
            'clip_idx': torch.tensor(clip_idx)
        }

        imgs, target = self._transforms(imgs, target)
        imgs = torch.stack(imgs, dim=0)
        return imgs, target

def make_coco_transforms(image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomResize([480], max_size=max_size),  # Fixed scaling
            normalize,
        ])
    
    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.actionvos_path)
    assert root.exists(), f'provided ActionVOS path {root} does not exist'
    ann_file = args.expression_file
    print('you are building actionvos {} set with {} , {}'.format(image_set, args.actionvos_path, ann_file))
    dataset = ActionVOSDatasetClipID(args.actionvos_path, ann_file, 
                              transforms=make_coco_transforms(image_set, max_size=args.max_size), 
                              return_masks=args.masks, num_frames=args.num_frames, 
                              max_skip=args.max_skip, use_weights=args.use_weights, 
                              image_set=image_set, index_map_path=args.index_map_path,
                              sample_map_path=args.sample_map_path)
    return dataset
# Select two parameters to the front
## 1. kernel  9/15/21##default is 9
## 2. noise_rate, whether to add noise to all### default is to add noise to all

## Input: annotations reading path, save path, corresponding expression reading path
## Output: annotation mask files with noise at the specified path  

import numpy as np
import os
import json
from PIL import Image
import cv2 as cv
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root_annotations_save', type=str, 
                    required=True,default="./dataset_visor_latest/Annotations_Sparse_noise")
parser.add_argument('--ksize', type=int, required=True,default=9)
parser.add_argument('--split', type=str, required=True, default='val_human')
parser.add_argument('--rate_noise', type=float, default=1.0)

def visualization_m_pil_noise(m_pil,m_pil_noise):
    # Create a canvas containing two subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Display the first image, ensuring the color palette is correctly applied
    ax[0].imshow(m_pil.convert('RGB'))
    ax[0].set_title('Original Annotations')
    ax[0].axis('off')  # Turn off axis
    # Display the second image, ensuring the color palette is correctly applied
    ax[1].imshow(m_pil_noise.convert('RGB'))
    ax[1].set_title('Noisy Annotations')
    ax[1].axis('off')  # Turn off axis
    # Adjust spacing between subplots
    plt.tight_layout()
    # Display images
    plt.show()

def generate_erosion_annotations(root_json,name_json,root_annotations_save,root_annotations,ksize = 9,rate_noise = 1.0): 
    path_json = os.path.join(root_json , name_json)
    with open(path_json,'r') as f:
        json1 = json.load(f)
    # json1['videos'].keys() traversal # file_name
    ############### e.g., # folder 00000004_P01_107_close_fridge
    # Traverse json1['videos']['00000004_P01_107_close_fridge']['frames']
    ############## # For frame_name specified in json file
    palette = Image.open('./annotations/00000.png').getpalette()

    for file_name,exp_frames in tqdm(json1['videos'].items()):
        exps = exp_frames['expressions']
        for frame in exp_frames['frames']:
            annotations_path = os.path.join(root_annotations,file_name,frame+'.png')
            annotations = np.array(Image.open(annotations_path))
            height,weight = annotations.shape 
            m_pil_org = Image.fromarray(annotations,mode='P')
            m_pil_org.putpalette(palette)
            mask_noise_ls = []

            annotations_noise = np.zeros((height, weight), dtype=np.uint8)
            for expr_idx,expr in exps.items():
                obj_id = int(expr['obj_id'])
                positive = expr['positive']
                mask = np.where(annotations==obj_id,1,0).astype(np.uint8)#(annotations==obj_id)
                if positive == 0:
                    mask = np.where(mask==1,obj_id,0).astype(np.uint8)
                    mask_noise_ls.append(mask)
                else:
                    if np.random.random() < rate_noise:
                        kernel = np.ones((ksize,ksize), np.uint8)#specify kernel size and data type uint8
                        mask_noise = cv.dilate(mask,kernel,iterations=1)
                        mask_noise = np.where(mask_noise==1,obj_id,0).astype(np.uint8)
                    else:
                        mask_noise = np.where(mask==1,obj_id,0).astype(np.uint8)#########no processing
                    for row in range(height):
                        for col in range(weight):
                            if annotations_noise[row][col]:#do not overwrite
                                continue
                            else:
                                annotations_noise[row][col] = mask_noise[row][col]
            for mask_noise in mask_noise_ls:
                for row in range(height):
                    for col in range(weight):
                        if annotations_noise[row][col]:#do not overwrite
                            continue
                        else:
                            annotations_noise[row][col] = mask_noise[row][col]
            annotations_noise = annotations_noise.astype(np.uint8)
            annotations_save_dir = os.path.join(root_annotations_save,file_name)
            if not os.path.exists(annotations_save_dir):
                os.makedirs(annotations_save_dir)
            if (annotations != annotations_noise).any():
                m_pil_noise = Image.fromarray(annotations_noise,mode='P')
                m_pil_noise.putpalette(palette)
                annotations_noise_save_path = os.path.join(annotations_save_dir,frame+'.png')
                m_pil_noise.save(annotations_noise_save_path)#noise added, save noisy annotations

                annotations_save_path = os.path.join(annotations_save_dir,frame+'_org.png')
                m_pil_org.save(annotations_save_path)#noise added, save original image
            else:
                annotations_save_path = os.path.join(annotations_save_dir,frame+'.png')
                m_pil_org.save(annotations_save_path)#no noise added

def main():
    args = parser.parse_args()
    root_annotations_save = args.root_annotations_save
    ksize = args.ksize
    split = args.split
    rate_noise = args.rate_noise
    print("root_annotations_save:",root_annotations_save)
    print("ksize:",ksize)
    print("split:",split)
    print("rate_noise:",rate_noise)

    ### Clean data path
    root_json = "./dataset_visor/ImageSets"
    name_json = split + "_meta_expressions_promptaction.json"
    # path_json = os.path.join(root_json , name_json)
    if '_' in split:
        split = split.split('_')[0]
    # Save annotations path
    root_annotations_save = os.path.join(root_annotations_save,split)

    # Read annotations path
    root_annotations = "./dataset_visor/Annotations_Sparse/"+split
    generate_erosion_annotations(root_json = root_json,name_json = name_json,\
                                 root_annotations_save = root_annotations_save,\
                                 root_annotations = root_annotations,ksize = ksize,\
                                 rate_noise = rate_noise)
if __name__ == '__main__':
    main()
##########
# python  generate_annotations_mask_noise.py --ksize 9 
# --root_annotations_save ./dataset_visor_latest/Annotations_Sparse_dilation_rate100 --split train --rate_noise 1.0

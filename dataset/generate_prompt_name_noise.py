import os
import json
import copy
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--actionvos_path_input', type=str, required=True)
parser.add_argument('--actionvos_output_path_noise', type=str, required=True)
parser.add_argument('--rate_noise', type=float, default=0.2)
args = parser.parse_args()
actionvos_path_input = args.actionvos_path_input##'./dataset_visor/ImageSets'#
actionvos_output_path_noise = args.actionvos_output_path_noise##'./dataset_visor/ImageSets_noise'
rate_noise = args.rate_noise###0.2####
print("actionvos_path_input = ", actionvos_path_input)
print("actionvos_output_path_noise = ", actionvos_output_path_noise)
print("rate_noise = ", rate_noise)

def get_class_id(root = "./dataset_visor/ImageSets"):
    # Get IDs of all categories
    name1 = "train_objects_category.json"
    name2 = "val_objects_category.json"
    json1 = json.load(open(os.path.join(root , name1)))
    json2 = json.load(open(os.path.join(root , name2)))
    dict_id2categoryls = {}#one class_id corresponds to one or more categories
    class_id_set1 = set()#class_id of training set
    ls1_keys = list(json1['videos'].keys())
    for keys_json_idx,keys_json in enumerate(ls1_keys):
        for key,value in json1['videos'][ls1_keys[keys_json_idx]]['objects'].items():
            class_id_set1 = class_id_set1 | set([value['class_id']])
            if value['class_id'] in dict_id2categoryls:
                tmp_ls = dict_id2categoryls[value['class_id']]
                tmp_ls.append(value['category'])
                tmp_ls = sorted(list(set(tmp_ls)))
                dict_id2categoryls[value['class_id']] = tmp_ls
            else:
                dict_id2categoryls[value['class_id']] = [value['category']]

    class_id_set2 = set()#class_id of validation set
    ls2_keys = list(json2['videos'].keys())
    for keys_json_idx,keys_json in enumerate(ls2_keys):
        for key,value in json2['videos'][ls2_keys[keys_json_idx]]['objects'].items():
            class_id_set2 = class_id_set2 | set([value['class_id']])
            if value['class_id'] in dict_id2categoryls:
                tmp_ls = dict_id2categoryls[value['class_id']]
                tmp_ls.append(value['category'])
                tmp_ls = sorted(list(set(tmp_ls)))
                dict_id2categoryls[value['class_id']] = tmp_ls
            else:
                dict_id2categoryls[value['class_id']] = [value['category']]

    class_id_ls = list(class_id_set1 | class_id_set2)
    print("tot class id number is {}".format(len(class_id_ls)))
    return class_id_ls
class_id_ls = get_class_id(root = actionvos_path_input)

def get_noise_class_id(label_org, label_ls):
    # Remove label_org from label_ls
    label_ls_rest = [label for label in label_ls if label != label_org]

    # Check if there are remaining labels
    if not label_ls_rest:
        raise ValueError("No labels left after removing the original label.")

    # Randomly select one from label_ls_rest as noise_label
    idx = np.random.randint(0, len(label_ls_rest))
    noise_label = label_ls_rest[idx]#random.choice(label_ls_rest)

    return noise_label

def get_classid2category_dict_from_category(noise_positive_file_path,classid2category_dict = {}):
    with open(noise_positive_file_path,'r') as f:
        josn_category = json.load(f)
    for file_fold,value_obj in josn_category['videos'].items():
        dict_obj = value_obj['objects']
        for obj_id,value in dict_obj.items():
            category = value['category']
            class_id = value['class_id']
            if class_id in classid2category_dict:
                tmp = copy.deepcopy(classid2category_dict[class_id])
                tmp.append(category)
                tmp = list(set(tmp))
                classid2category_dict[class_id] = tmp
            else:
                classid2category_dict[class_id] = [category]
    return classid2category_dict

def get_train_val_dict():
    root_dir = "./dataset_visor"
    noise_positive = "ImageSets"
    train_json_name = "train_objects_category.json"#read category file
    noise_positive_file_path_train = os.path.join(root_dir,noise_positive,train_json_name)
    val_json_name = "val_objects_category.json"#read category file
    noise_positive_file_path_val = os.path.join(root_dir,noise_positive,val_json_name)
    classid2category_dict = get_classid2category_dict_from_category(noise_positive_file_path_train)
    classid2category_dict = get_classid2category_dict_from_category(noise_positive_file_path_val,classid2category_dict = classid2category_dict)
    return classid2category_dict


def generate_noise_label_json(json1,json2,rate_noise,split = 'train'):
    classid2category_dict_val_train = get_train_val_dict()
    tot1_times = 0
    tot2_times = 0
    tot_times = 0
    path_val_narration = f"./dataset_visor/ImageSets/{split}.json"#contains annotations, verbs, val.json or train.json##only process train
    with open(path_val_narration,'r') as f:
        json_val_narration = json.load(f)
    for value_narration,json1_keys in tqdm(zip(json_val_narration,list(json1['videos'].keys()))):
        narration = value_narration['narration']
        object_classes = value_narration['object_classes']
        tmp_json_1 = copy.deepcopy(json1['videos'][json1_keys]['expressions'])

        tmp_json_2 = copy.deepcopy(json2['videos'][json1_keys]['objects'])

        for key,value in tmp_json_1.items():
            tot_times = tot_times + 1
            obj_id = value['obj_id']
            exp_obj_id = str(int(obj_id)-1)
            #### Add noise
            class_id_org = value['class_id']
            value['class_id_org'] = class_id_org
            tmp_json_2[obj_id]['class_id_org'] = class_id_org
            tmp_json_1[exp_obj_id]['exp_org'] = tmp_json_1[exp_obj_id]['exp']
            tmp_json_2[obj_id]['category_org'] = tmp_json_2[obj_id]['category']
            if np.random.random() < rate_noise:
                tot1_times = tot1_times + 1
                noise_class_id = get_noise_class_id(value['class_id'],class_id_ls)# randomly select one from remaining classes
                value['class_id'] = noise_class_id
                tmp_json_2[obj_id]['class_id'] = noise_class_id
                category_ls = classid2category_dict_val_train[noise_class_id]
                category_ls_len = len(category_ls)
                if category_ls_len == 1:
                    category_noise = category_ls[0]
                else:
                    idx_noise = np.random.randint(category_ls_len)#randomly generate a number from 0,1,...,category_ls_len-1
                    category_noise = category_ls[idx_noise]
                tmp_json_2[obj_id]['category'] = category_noise
                noise_prompt_name = category_noise + " used in the action of "+ narration
                tmp_json_1[exp_obj_id]['exp'] = noise_prompt_name
        json1['videos'][json1_keys]['expressions'] = tmp_json_1
        json2['videos'][json1_keys]['objects'] = tmp_json_2
    print("tot times = {}, tot class_id times = {}, tot positive times = {}".format(tot_times,tot1_times,tot2_times))
    return json1,json2
def main():
    json_root = actionvos_path_input
    print("json_root = ", json_root)
    ###################################################################
    ############### train ###                                   
    json1_name_input = "train_meta_expressions_promptaction.json"
    json1_name_output = "train_meta_expressions_promptaction.json"
    json1 = json.load(open(os.path.join(json_root , json1_name_input)))

    json2_name_input = "train_objects_category.json"
    json2_name_output = "train_objects_category.json"
    json2 = json.load(open(os.path.join(json_root , json2_name_input)))
    
    json_root = actionvos_output_path_noise
    os.makedirs(json_root,exist_ok=True)
    json1_noise,json2_noise = generate_noise_label_json(json1,json2,rate_noise)
    json1_noise = json.dumps(json1_noise)
    json2_noise = json.dumps(json2_noise)
    with open(os.path.join(json_root, json1_name_output), "w") as f:
        f.write(json1_noise)
    with open(os.path.join(json_root, json2_name_output), "w")  as f:
        f.write(json2_noise)

if __name__ == "__main__":
    main()

#  python generate_prompt_name_noise.py  --actionvos_path_input ./dataset_visor_latest_0406/ImageSets  \
#  --actionvos_output_path_noise ./dataset_visor_latest_0406/ImageSets_noise  \
#   --rate_noise 0.4

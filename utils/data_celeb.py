import json
import os
from collections import Counter
from tqdm import tqdm

import cv2
import torch
from PIL import Image
from torchvision import transforms
from torch.nn.functional import interpolate
import numpy as np

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--base_dir", type=str, default="/export/home1/NoCsBack/working/CelebTo")
parser.add_argument("--img_dir_name", type=str, default="images_ct")
parser.add_argument("--avg_face_dir", type=str, default="avg_faces_allname")
parser.add_argument("--dict_dir", type=str, default="/cw/liir_code/NoCsBack/tingyu/FaceNaming/CelebrityTo")
parser.add_argument("--dict_name", type=str, default="celeb_dict.json")

parser.add_argument("--out_dir", type=str, default="/export/home1/NoCsBack/working/tingyu/face_naming/celeb")

parser.add_argument("--alpha", type=float, default=0.15)
parser.add_argument("--agree_type", type=str, default="diag")

parser.add_argument("--data_name", type=str, default="allname")


args = parser.parse_args()


def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data

def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = cv2.resize(
            img,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = imresample(
            img.permute(2, 0, 1).unsqueeze(0).float(),
            (image_size, image_size)
        ).byte().squeeze(0).permute(1, 2, 0)
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out

def count_name_occur(dict_dir, dict_name):
    name_list = []
    # ner_list = []
    with open(os.path.join(dict_dir, dict_name)) as f:
        data_dict = json.load(f)

    for key in tqdm(data_dict.keys()):
        for name in data_dict[key]["name_list"]:
            name_list.append(name)
    return Counter(name_list)


def make_unique_name_dict(base_dir, dict_name):
    name_counts = count_name_occur(base_dir, dict_name)

    with open(os.path.join(base_dir, dict_name)) as f:
        data_dict = json.load(f)

    dict_out = {}
    for name in tqdm(name_counts.keys()):
        for sample in data_dict.values():
            if name in sample["name_list"] and name != "NONAME":
                name_pos = sample["name_list"].index(name)
                if name not in dict_out.keys():
                    dict_out[name] = {}
                
                if "img_dir" not in dict_out[name].keys():
                    dict_out[name]["img_dir"] = [sample["img_dir"]]
                    dict_out[name]["bbox"] = [sample["bbox"][name_pos]]
                else:
                    dict_out[name]["img_dir"].append(sample["img_dir"])
                    dict_out[name]["bbox"].append(sample["bbox"][name_pos])
    return dict_out



def average_img_3(imlist, face_bboxes):
    # Alternative method using numpy mean function
    # images = np.array([np.array(Image.open(fname)) for fname in imlist])
    images = np.array(
        [np.array(crop_resize(Image.open(fname).convert("RGB"), face_bbox, 160)) for fname, face_bbox in zip(imlist, face_bboxes)]
    )

    arr = np.array(np.mean(images, axis=(0)), dtype=np.uint8)
    out = Image.fromarray(arr)
    return out

def make_avg_face_one_one(base_dir, unique_name_dict, avg_face_dir):
    unique_name_avg_face_dict = {}
    for name, faces in tqdm(unique_name_dict.items()):
        out_face_dir = os.path.join(avg_face_dir, "{}.png".format(name))
        unique_name_avg_face_dict[name] = {}
        unique_name_avg_face_dict[name]["avg_face_dir"] = out_face_dir

        if len(faces["img_dir"]) == 1:
            img_dir = os.path.join(base_dir, faces["img_dir"][0])
            img = Image.open(img_dir).convert("RGB")
            face_bbox = faces["bbox"][0]
            crop_face = crop_resize(img, face_bbox, 160)
            print(type(crop_face))
            img.close()
            crop_face.save(out_face_dir)

        else:
            face_bboxes = []
            img_dirs = []
            for i in range(len(faces["img_dir"])):
                img_dirs.append(os.path.join(base_dir, faces["img_dir"][i]))
                face_bboxes.append(faces["bbox"][i])

            avg_face = average_img_3(img_dirs, face_bboxes)
            avg_face.save(out_face_dir)
    return unique_name_avg_face_dict


def find_aligned_names(sim_lists, name_list):
    aligned_names = []
    for sim_list in sim_lists:
        if max(sim_list) > 0:
            aligned_name = name_list[sim_list.index(max(sim_list))]
            aligned_names.append(aligned_name[0])
        else:
            aligned_names.append("NONAME")
    return aligned_names


def unique(list1):
    # initialize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def duplicates_idx(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def remove_multiple_from_list(tgt_list, remove_indices):
    removed_items = []
    remained_items = []
    for i in range(len(tgt_list)):
        if i in remove_indices:
            removed_items.append(tgt_list[i])
        else:
            remained_items.append(tgt_list[i])
    return remained_items, removed_items


def del_matched_pair_from_dict(rest_dict, rest_result_dict, unique_name_dict):
    """
    Create data split using heuristic: only use unmatched names for stage2 training
    :param rest_dict: data dict for data_one_rest
    :param rest_result_dict: result_dict for data_one_rest from one-one model
    :param unique_name_dict: dict of unique names
    :return: matched_dict: sub_dict of rest_result_dict containing all matched names & faces
             unmatched_dict: sub_dict of rest_result_dict containing the rest
             The two will be used together for evaluation
    """
    unique_names = unique_name_dict.keys()

    matched_dict = {}
    unmatched_dict = {}

    for key, sample in tqdm(rest_dict.items()):
        results_key = os.path.join("/cw/working-rose/CelebTo/images_ct", sample["img_dir"])

        results_name_list = rest_result_dict[results_key]["name_list"]
        sim_scores = rest_result_dict[results_key]["sim_face_name"]

        aligned_names = find_aligned_names(sim_scores, results_name_list)
        # aligned_names_unique = list(set(aligned_names))
        aligned_names_unique = unique([name for name in aligned_names])

        deleted_face_idx = []
        deleted_name_idx = []

        name_from_one_one = []
        for i in range(len(results_name_list)):
            for name in results_name_list[i]:
                if name in unique_names:
                    name_from_one_one.append(name)
        if len(name_from_one_one) == 0:
            unmatched_dict[key] = {}
            unmatched_dict[key]["img_dir"] = sample["img_dir"]
            unmatched_dict[key]["name_id"] = sample["name_id"]
            unmatched_dict[key]["bbox"] = sample["bbox"]
            unmatched_dict[key]["face_id"] = sample["face_id"]
            unmatched_dict[key]["name_list"] = sample["name_list"]
            unmatched_dict[key]["gt_link"] = sample["gt_link"]
        else:
            for idx, name in enumerate(name_from_one_one):

                if name in aligned_names_unique:
                    name_aligned_count = Counter([name for name in aligned_names])
                    name_index = results_name_list.index([name])

                    aligned_idx = duplicates_idx(aligned_names, name)

                    aligned_sim_scores = [sim_scores[i] for i in aligned_idx]
                    name_associate_scores = [score[name_index] for score in aligned_sim_scores]

                    if max(name_associate_scores) < 0:
                        continue
                    elif name_aligned_count[name] > 1:
                        # find max sim score at the index
                        max_index = name_associate_scores.index(max(name_associate_scores))

                        face_start_index = sample["bbox"].count(-1)
                        deleted_face_idx.append(aligned_idx[max_index] + face_start_index)
                        deleted_name_idx.append(name_index)
                    else:
                        face_start_index = sample["bbox"].count(-1)

                        deleted_face_idx.append(aligned_names.index(name) + face_start_index)
                        deleted_name_idx.append(results_name_list.index([name]))

        if len(deleted_face_idx) > 0:
            matched_dict[key] = {}
            unmatched_dict[key] = {}
            matched_dict[key]["img_dir"] = sample["img_dir"]
            unmatched_dict[key]["img_dir"] = sample["img_dir"]

            bbox_left, bbox_removed = remove_multiple_from_list(sample["bbox"], deleted_face_idx)
            gt_link_left, gt_link_removed = remove_multiple_from_list(sample["gt_link"], deleted_face_idx)
            name_list_left, name_list_removed = remove_multiple_from_list(sample["name_list"], deleted_name_idx)

            matched_dict[key]["bbox"] = bbox_removed
            matched_dict[key]["gt_link"] = gt_link_removed
            matched_dict[key]["name_list"] = name_list_removed

            unmatched_dict[key]["bbox"] = bbox_left
            unmatched_dict[key]["gt_link"] = gt_link_left
            unmatched_dict[key]["name_list"] = name_list_left
            unmatched_dict[key]["name_id"] = sample["name_id"]
            unmatched_dict[key]["face_id"] = sample["face_id"]
        else:
            unmatched_dict[key] = {}
            unmatched_dict[key]["img_dir"] = sample["img_dir"]
            unmatched_dict[key]["name_id"] = sample["name_id"]
            unmatched_dict[key]["bbox"] = sample["bbox"]
            unmatched_dict[key]["face_id"] = sample["face_id"]
            unmatched_dict[key]["name_list"] = sample["name_list"]
            unmatched_dict[key]["gt_link"] = sample["gt_link"]

    return matched_dict, unmatched_dict


def get_easy_subset_2name(celeb_dict_2name, celeb_dict):
    """get subset of 2name's with one NONAME and one name
    """
    celeb_dict_2name_easy = {}
    celeb_dict_2name_easy_rest = {}
    easy_keys = []
    for key, sample in tqdm(celeb_dict_2name.items()):
        if "NONAME" in sample["name_list"]:
            celeb_dict_2name_easy[key] = {}
            celeb_dict_2name_easy[key] = sample
            easy_keys.append(key)
        else:
            continue
    
    for key, sample in tqdm(celeb_dict.items()):
        if key not in easy_keys:
            celeb_dict_2name_easy_rest[key] = {}
            celeb_dict_2name_easy_rest[key] = sample
        else:
            continue

    return celeb_dict_2name_easy, celeb_dict_2name_easy_rest


def get_subset_2name_no_noname(celeb_dict_2name, celeb_dict):
    """get subset of 2name's without NONAME
    """
    celeb_dict_2name_allname = {}
    celeb_dict_2name_allname_rest = {}
    easy_keys = []
    for key, sample in tqdm(celeb_dict_2name.items()):
        if "NONAME" not in sample["name_list"]:
            celeb_dict_2name_allname[key] = {}
            celeb_dict_2name_allname[key] = sample
            easy_keys.append(key)
        else:
            continue
    
    for key, sample in tqdm(celeb_dict.items()):
        if key not in easy_keys:
            celeb_dict_2name_allname_rest[key] = {}
            celeb_dict_2name_allname_rest[key] = sample
        else:
            continue

    return celeb_dict_2name_allname, celeb_dict_2name_allname_rest


if __name__ == "__main__":
    with open(os.path.join(args.dict_dir, args.dict_name)) as f:
        data_celeb_full = json.load(f)

    data_celeb_2name = {}
    data_celeb_2name_rest = {}

    for key in tqdm(data_celeb_full.keys()):
        if len(data_celeb_full[key]["name_list"]) < 3:
            data_celeb_2name[key] = {}
            data_celeb_2name[key] = data_celeb_full[key]
        else:
            data_celeb_2name_rest[key] = {}
            data_celeb_2name_rest[key] = data_celeb_full[key]

    print(f"size of the full dataset:{len(data_celeb_full)}")
    print(f"size of subset with less than 3 names:{len(data_celeb_2name)}")
    print(f"size of subset with at least 3 names:{len(data_celeb_2name_rest)}")

    # with open(os.path.join(args.dict_dir, "celeb_dict_2name.json"), "w") as f:
    #     json.dump(data_celeb_2name, f)

    # with open(os.path.join(args.dict_dir, "celeb_dict_2name_rest.json"), "w") as f:
    #     json.dump(data_celeb_2name_rest, f)

    # unique_name_dict = make_unique_name_dict(args.dict_dir, "celeb_dict_2name.json")

    # with open(os.path.join(args.dict_dir, "celeb_dict_2name_unique.json"), "w") as f:
    #     json.dump(unique_name_dict, f)


    # data_celeb_2name_allname, data_celeb_2name_allname_rest = get_subset_2name_no_noname(data_celeb_2name, data_celeb_full)

    # print(f"size of easy subset with 2 names without NONAME:{len(data_celeb_2name_allname)}")
    # print(f"size of complementary subset:{len(data_celeb_2name_allname_rest)}")

    # with open(os.path.join(args.dict_dir, "celeb_dict_2name_allname.json"), "w") as f:
    #         json.dump(data_celeb_2name_allname, f)
        
    # with open(os.path.join(args.dict_dir, "celeb_dict_2name_allname_rest.json"), "w") as f:
    #     json.dump(data_celeb_2name_allname_rest, f)
    

    # unique_name_dict_allname = make_unique_name_dict(args.dict_dir, "celeb_dict_2name_allname.json")

    # with open(os.path.join(args.dict_dir, "celeb_dict_2name_unique_allname.json"), "w") as f:
    #     json.dump(unique_name_dict_allname, f)

    # print(len(unique_name_dict_allname))


    # unique_name_dict_easy = make_unique_name_dict(args.dict_dir, "celeb_dict_2name_easy.json")

    # with open(os.path.join(args.dict_dir, "celeb_dict_2name_unique_easy.json"), "w") as f:
    #     json.dump(unique_name_dict_easy, f)

    # print(len(unique_name_dict_easy))

    img_dir = os.path.join(args.base_dir, args.img_dir_name)
    avg_face_dir = os.path.join(args.base_dir, args.avg_face_dir)

    # unique_name_avg_face_dict = make_avg_face_one_one(img_dir, unique_name_dict, avg_face_dir)

    # with open(os.path.join(args.dict_dir, "celeb_dict_2name_avg_face.json"), "w") as f:
    #     json.dump(unique_name_avg_face_dict, f)

    # unique_name_easy_avg_face_dict = make_avg_face_one_one(img_dir, unique_name_dict_easy, avg_face_dir)

    # with open(os.path.join(args.dict_dir, "celeb_dict_2name_avg_face_easy.json"), "w") as f:
    #     json.dump(unique_name_easy_avg_face_dict, f)

    # unique_name_allname_avg_face_dict = make_avg_face_one_one(img_dir, unique_name_dict_allname, avg_face_dir)

    # with open(os.path.join(args.dict_dir, "celeb_dict_2name_avg_face_allname.json"), "w") as f:
    #     json.dump(unique_name_allname_avg_face_dict, f)

    # with open(os.path.join(args.dict_dir, "celeb_dict_2name_avg_face_easy.json")) as f:
    #     unique_name_easy_avg_face_dict = json.load(f)

    # unique_name_avg_face_dict_update = {}

    # for key in unique_name_easy_avg_face_dict.keys():
    #     unique_name_avg_face_dict_update[key] = {}
    #     unique_name_avg_face_dict_update[key]["avg_face_dir"] = unique_name_easy_avg_face_dict[key]["avg_face_dir"].replace("/export/home1/NoCsBack/working/CelebTo", "/cw/working-rose/CelebTo")

    # with open(os.path.join(args.dict_dir, "celeb_dict_2name_avg_face_easy.json"), "w") as f:
    #     json.dump(unique_name_avg_face_dict_update, f)



    # with open(os.path.join(args.dict_dir, "celeb_dict_2name_unique.json")) as f:
    #     unique_name_dict = json.load(f)

    # unique_name_dict_new = {}
    # for key,sample in unique_name_dict.items():
    #     if key != "NONAME":
    #         unique_name_dict_new[key] = {}
    #         unique_name_dict_new[key] = sample




    with open(os.path.join(args.dict_dir, "celeb_dict_2name_unique_allname.json")) as f:
        unique_name_dict_new = json.load(f)

    with open(os.path.join(args.dict_dir, "celeb_dict_2name_allname_rest.json")) as f:
        celeb_dict_2name_rest = json.load(f)

    with open(os.path.join(args.out_dir, "rest_results_2name_allname_alpha0.15_agree-diag_new.json")) as f:
        celeb_2name_rest_result = json.load(f)

    matched_dict, unmatched_dict = del_matched_pair_from_dict(celeb_dict_2name_rest, celeb_2name_rest_result, unique_name_dict_new)

    print(len(unmatched_dict))

    # with open(os.path.join(args.dict_dir, "rest_unmatch_2name_{}_alpha{}_agree-{}_new.json".format(args.data_name, args.alpha, args.agree_type)), "w") as f:
    #         json.dump(unmatched_dict, f)
        
    # with open(os.path.join(args.dict_dir, "rest_match_2name_{}_alpha{}_agree-{}_new.json".format(args.data_name, args.alpha, args.agree_type)), "w") as f:
    #     json.dump(matched_dict, f)


    # make easy dict

    # data_celeb_2name_easy, data_celeb_2name_easy_rest = get_easy_subset_2name(data_celeb_2name, data_celeb_full)

    # print(f"size of easy subset with 2 names including 1 NONAME:{len(data_celeb_2name_easy)}")
    # print(f"size of complementary subset:{len(data_celeb_2name_easy_rest)}")

    # with open(os.path.join(args.dict_dir, "celeb_dict_2name_easy.json"), "w") as f:
    #         json.dump(data_celeb_2name_easy, f)
        
    # with open(os.path.join(args.dict_dir, "celeb_dict_2name_easy_rest.json"), "w") as f:
    #     json.dump(data_celeb_2name_easy_rest, f)

   
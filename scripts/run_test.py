import json
import os
import re
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--sys_dir", type=str, default="/cw/working-rose/tingyu/FaceNaming")
parser.add_argument("--base_dir_name", type=str, default="Berg")
parser.add_argument("--out_dir", type=str, default="/export/home2/NoCsBack/working/tingyu/face_naming/unsup_frag")
parser.add_argument("--result_json_name", type=str, default="unsup_frag_ame_two-proj_dim:128_biasTrue_1.0data:train_loss:batch-0.25-agree_bsz:40_epoch30_op:adam_lr0.0003_contextFalse.json")
parser.add_argument("--data_type", type=str, default="all")
parser.add_argument("--data_dict", type=str, default="gt_dict_cleaned_phi_face_name.json")
parser.add_argument("--add_noname", type=bool, default=False)

args = parser.parse_args()


def match_noface_from_dict(img_name, data_dict):
    """
    match face_x from data_dict using img_name
    """
    face_x = []
    name_list = []
    ner_list = []
    for values in data_dict.values():
        if values['img_name'] == [img_name]:
            face_x = values["face_x"]
            name_list = values["name_list"]
            ner_list = values['ner']
        else:
            continue
    return face_x, name_list, ner_list


def make_gt_pred_list(face_x, sim_list, name_list, ner_list, add_noname):
    """
    make ground truty & predited name-face alignment list
    """
    gt_list = []
    num_names = len(face_x)
    noface_counter = 0

    for i in range(num_names):
        if face_x[i] == -1 and ner_list[i] != "NOFACEWRONG":
            gt_list.append("NOFACE")
            noface_counter += 1
        elif face_x[i] == -1 and ner_list[i] == "NOFACEWRONG":
            gt_list.append("WRONGFACE")
            noface_counter += 1
        elif ner_list[i] == "NONAMEWRONG":
            gt_list.append("WRONGNAME")
        else:
            gt_list.append(name_list[i][0])

    pred_list = ["NOFACE"] * num_names  # if add_noname, len(name_list) = num_names + 1

    if add_noname:
        for j in range(len(sim_list)):
            if max(sim_list[j]) > 0 and sim_list[j].index(max(sim_list[j])) < len(name_list):
                pred_list[j + noface_counter] = name_list[sim_list[j].index(max(sim_list[j]))][0]
            else:
                pred_list[j + noface_counter] = "NONAME"

    else:
        for j in range(len(sim_list)):
            if max(sim_list[j]) > 0:
                pred_list[j + noface_counter] = name_list[sim_list[j].index(max(sim_list[j]))][0]
            else:
                pred_list[j + noface_counter] = "NONAME"

    return gt_list, pred_list



def make_gt_pred_list_celeb(sim_list, name_list, gt_link):
    """
    make ground truty & predited name-face alignment list for CelebTo data
    """
    gt_list = []
    pred_list = []

    for i in range(len(gt_link)):
        gt_list.append(gt_link[i][0][0])

    for j in range(len(sim_list)):
        if max(sim_list[j]) > 0 and sim_list[j].index(max(sim_list[j])) < len(name_list):
            pred_list.append(name_list[sim_list[j].index(max(sim_list[j]))][0])
        else:
            pred_list.append("NONAME")

    return gt_list, pred_list



def compare_gt_pred_list(gt_list, pred_list):
    """
    compare ground truty & predited name-face alignment list
    """
    gt_count = 0
    pred_count = 0
    pred_true_count = 0

    for i in range(len(gt_list)):
        if gt_list[i].startswith("WRONG"):
            pred_count += 1
        elif gt_list[i] == pred_list[i]:
            gt_count += 1
            pred_count += 1
            pred_true_count += 1
        else:
            gt_count += 1
            pred_count += 1

    return gt_count, pred_count, pred_true_count


def cal_f1_json_noface_noname(data_dict, results_json, add_noname):
    """
    evaluate performance according to dict of data,
    we rely on -1 in face_x to find NOFACE
    :param data_dict: dict of training data
    :param results_json: dict containing similarity scores
    :return:
    """
    all_pred_count = 0
    all_gt_count = 0
    all_pred_true_count = 0

    for index, key in enumerate(results_json):
        img_name = key
        face_x, name_list, ner_list = match_noface_from_dict(img_name, data_dict)
        sim_list = results_json[key]["sim_face_name"]

        gt_list, pred_list = make_gt_pred_list(face_x, sim_list, name_list, ner_list, add_noname)

        gt_count, pred_count, pred_true_count = compare_gt_pred_list(gt_list, pred_list)
        all_gt_count += gt_count
        all_pred_count += pred_count
        all_pred_true_count += pred_true_count

    precision = all_pred_true_count / all_pred_count
    recall = all_pred_true_count / all_gt_count
    f1 = 2 * precision * recall / (precision + recall)
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def cal_f1_json_noface_noname_noneg(data_dict, results_json, add_noname):
    """
    evaluate performance according to dict of data,
    we rely on -1 in face_x to find NOFACE
    :param data_dict: dict of training data
    :param results_json: dict containing similarity scores
    :return:
    """
    all_pred_count = 0
    all_gt_count = 0
    all_pred_true_count = 0

    for index, key in enumerate(results_json):
        img_name = key
        face_x, name_list, ner_list = match_noface_from_dict(img_name, data_dict)
        sim_list = results_json[key]["sim_face_name"]

        gt_list, pred_list = make_gt_pred_list_noneg(face_x, sim_list, name_list, ner_list, add_noname)

        gt_count, pred_count, pred_true_count = compare_gt_pred_list(gt_list, pred_list)
        all_gt_count += gt_count
        all_pred_count += pred_count
        all_pred_true_count += pred_true_count

    precision = all_pred_true_count / all_pred_count
    recall = all_pred_true_count / all_gt_count
    f1 = 2 * precision * recall / (precision + recall)
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }



def make_gt_pred_list_noneg(face_x, sim_list, name_list, ner_list, add_noname):
    """
    make ground truty & predited name-face alignment list
    """
    gt_list = []
    num_names = len(face_x)
    noface_counter = 0

    for i in range(num_names):
        if face_x[i] == -1 and ner_list[i] != "NOFACEWRONG":
            gt_list.append("NOFACE")
            noface_counter += 1
        elif face_x[i] == -1 and ner_list[i] == "NOFACEWRONG":
            gt_list.append("WRONGFACE")
            noface_counter += 1
        elif ner_list[i] == "NONAMEWRONG":
            gt_list.append("WRONGNAME")
        else:
            gt_list.append(name_list[i][0])

    pred_list = ["NOFACE"] * num_names  # if add_noname, len(name_list) = num_names + 1

    if add_noname:
        for j in range(len(sim_list)):
            if sim_list[j].index(max(sim_list[j])) < len(name_list):
                pred_list[j + noface_counter] = name_list[sim_list[j].index(max(sim_list[j]))][0]
            else:
                pred_list[j + noface_counter] = "NONAME"

    else:
        for j in range(len(sim_list)):
            pred_list[j + noface_counter] = name_list[sim_list[j].index(max(sim_list[j]))][0]

    return gt_list, pred_list


def cal_f1_json_celeb(results_json, out_dir, out_file_name):
    """
    evaluate performance according to dict of data,
    we rely on -1 in face_x to find NOFACE
    :param results_json: dict containing similarity scores
    :return:
    """
    all_pred_count = 0
    all_gt_count = 0
    all_pred_true_count = 0

    align_result_dict = {}

    for _, key in tqdm(enumerate(results_json)):
        name_list = results_json[key]["name_list"]
        sim_list = results_json[key]["sim_face_name"]
        gt_link = results_json[key]["gt_link"]
        align_result_dict[key] = {}
        align_result_dict[key]["name_list"] = name_list

        gt_list, pred_list = make_gt_pred_list_celeb(sim_list, name_list, gt_link)
        align_result_dict[key]["pred_list"] = pred_list

        gt_count, pred_count, pred_true_count = compare_gt_pred_list(gt_list, pred_list)
        all_gt_count += gt_count
        all_pred_count += pred_count
        all_pred_true_count += pred_true_count

    precision = all_pred_true_count / all_pred_count
    recall = all_pred_true_count / all_gt_count
    f1 = 2 * precision * recall / (precision + recall)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def cal_f1_json_celeb_noneg(results_json, out_dir, out_file_name):
    """
    evaluate performance according to dict of data,
    we rely on -1 in face_x to find NOFACE
    :param results_json: dict containing similarity scores
    :return:
    """
    all_pred_count = 0
    all_gt_count = 0
    all_pred_true_count = 0

    align_result_dict = {}

    for _, key in tqdm(enumerate(results_json)):
        name_list = results_json[key]["name_list"]
        sim_list = results_json[key]["sim_face_name"]
        gt_link = results_json[key]["gt_link"]
        align_result_dict[key] = {}
        align_result_dict[key]["name_list"] = name_list

        gt_list, pred_list = make_gt_pred_list_celeb_noneg(sim_list, name_list, gt_link)
        align_result_dict[key]["pred_list"] = pred_list

        gt_count, pred_count, pred_true_count = compare_gt_pred_list(gt_list, pred_list)
        all_gt_count += gt_count
        all_pred_count += pred_count
        all_pred_true_count += pred_true_count

    precision = all_pred_true_count / all_pred_count
    recall = all_pred_true_count / all_gt_count
    f1 = 2 * precision * recall / (precision + recall)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def make_gt_pred_list_celeb_noneg(sim_list, name_list, gt_link):
    """
    make ground truty & predited name-face alignment list for CelebTo data
    does not consider negative situation
    """
    gt_list = []
    pred_list = []

    for i in range(len(gt_link)):
        gt_list.append(gt_link[i][0][0])

    for j in range(len(sim_list)):
        if sim_list[j].index(max(sim_list[j])) < len(name_list):
            pred_list.append(name_list[sim_list[j].index(max(sim_list[j]))][0])
        else:
            pred_list.append("NONAME")

    return gt_list, pred_list


if __name__ == "__main__":
    if args.add_noname:
        print(args.add_noname)

    base_dir = os.path.join(args.sys_dir, args.base_dir_name)
    out_dir = args.out_dir
    with open(os.path.join(out_dir, args.result_json_name)) as f:
        result_json = json.load(f)

    with open(os.path.join(base_dir, args.data_dict)) as f:
        data_dict = json.load(f)

    print("data type is:{}".format(args.data_type))
    print(args.result_json_name)
    if args.data_type == "celeb":
        out_file_name = "align" + args.result_json_name
        print(cal_f1_json_celeb(result_json, out_dir, out_file_name))
    elif args.data_type == "celeb_noneg":
        out_file_name = "align" + args.result_json_name
        print(cal_f1_json_celeb_noneg(result_json, out_dir, out_file_name))
    elif args.data_type == "noname_noneg":
        print(cal_f1_json_noface_noname_noneg(data_dict, result_json, args.add_noname))
    else:
        print(cal_f1_json_noface_noname(data_dict, result_json, args.add_noname))




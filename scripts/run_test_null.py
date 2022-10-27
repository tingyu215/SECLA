import json
import os
import argparse
from run_test import match_noface_from_dict, make_gt_pred_list_noneg, compare_gt_pred_list

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_type", type=str, default="celeb")
parser.add_argument("--sys_dir", type=str, default="/cw/liir_code/NoCsBack/tingyu/FaceNaming")
parser.add_argument("--base_dir_name", type=str, default="Berg")
parser.add_argument("--out_dir", type=str, default="/cw/working-sauron/tingyu/face_naming/celeb_incre")
# parser.add_argument("--result_json_name", type=str, default="unsup_frag__name_two5-proj_dim:128_biasTrue_1.0data:train_loss:batch-0.15-agree-normal-full_bsz:20_shuffle-True_epoch30_op:adam_lr0.0003_nonameTrue_True_textModelbert-uncased_finetune-False_mean-True-True-layerS-4.json")
parser.add_argument("--result_json_name", type=str, default="stage1:agreediag_alpha0.15_bsz20-2_splitTrue_False_add-samplerandom_False_add-tounmatch_make_newTrue-both_face_conTrue-both-False-False_add_d_one-True-False_allname_nullface-True.json")

parser.add_argument("--data_type", type=str, default="noname_noneg")
parser.add_argument("--data_dict", type=str, default="gt_dict_cleaned_phi_face_name.json")
parser.add_argument("--add_noname", default=False, type=lambda x: (str(x).lower() == 'true'))
args = parser.parse_args()


def cal_f1_json_noface_noname_noneg_delete_noname_noface(data_dict, results_json, noname_dict, noface_dict, add_noname):
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
        # print(gt_list, pred_list)

        gt_count, pred_count, pred_true_count = compare_gt_pred_list(gt_list, pred_list)
        all_gt_count += gt_count
        all_pred_count += pred_count
        all_pred_true_count += pred_true_count
    
    precision = all_pred_true_count / all_pred_count
    recall = all_pred_true_count / all_gt_count
    f1 = 2 * precision * recall / (precision + recall)

    noname_all_count, noname_acc_count =cal_acc_noname(noname_dict)
    precision_noname = (all_pred_true_count-noname_acc_count) / (all_pred_count - noname_all_count)
    recall_noname = (all_pred_true_count-noname_acc_count) / (all_gt_count - noname_all_count)
    f1_noname = 2 * precision_noname * recall_noname / (precision_noname + recall_noname)

    # noface_all_count, noface_acc_count = cal_acc_noface(noface_dict)
    # precision_noface = (all_pred_true_count - noface_acc_count) / (all_pred_count - noface_all_count)
    # recall_noface = (all_pred_true_count - noface_acc_count) / (all_gt_count - noface_all_count)
    # f1_noface = 2 * precision_noface * recall_noface / (precision_noface + recall_noface)


    # precision_noboth = (all_pred_true_count-noname_acc_count-noface_acc_count) / (all_pred_count-noname_all_count - noface_all_count)
    # recall_noboth = (all_pred_true_count-noname_acc_count-noface_acc_count) / (all_gt_count-noname_all_count - noface_all_count)
    # f1_noboth = 2 * precision_noboth * recall_noboth / (precision_noboth + recall_noboth)
    
    # print(all_pred_true_count, noname_acc_count, noface_acc_count, all_gt_count,noname_all_count, noface_all_count, all_pred_count)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Precision_noname": precision_noname,
        "Recall_noname": recall_noname,
        "F1_noname": f1_noname,
        # "Precision_noface": precision_noface,
        # "Recall_noface": recall_noface,
        # "F1_noface": f1_noface,
        # "Precision_noboth": precision_noboth,
        # "Recall_noboth": recall_noboth,
        # "F1_noboth": f1_noboth,
    }

def gen_subdict_noname(result_dict):
    out_dict = {}
    for key in result_dict.keys():
        noname_indices, noname_indices_sim, noface_indices = get_noname_indices(result_dict[key]["gt_link"])
        # print(noname_indices, noname_indices_sim)
        if noname_indices != [] and noname_indices_sim != []:
            if len(result_dict[key]["gt_link"]) != len(result_dict[key]["sim_face_name"]) + len(noface_indices):
                print(f"check key1:{key}")
                # print(result_dict[key]["gt_link"])
                # print(result_dict[key]["sim_face_name"])
            elif key not in out_dict.keys():
                out_dict[key] = {}
                out_dict[key]["noname_sim_idx"] = noname_indices_sim
                out_dict[key]["gt_link_full"] = result_dict[key]["gt_link"]
                out_dict[key]["sim_face_name_full"] = result_dict[key]["sim_face_name"]
            else:
                print(f"check key:{key}")
        else:
            continue
    return out_dict

def indices(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

def get_noname_indices(gt_link_list):
    linked_list = [links[1] for links in gt_link_list]
    noface_indices = indices(linked_list, ["NOFACE"])
    # noface_indices = indices([links[0] for links in gt_link_list], ["NOFACE"])
    if ["NOFACE"] in linked_list and ["NONAME"] in linked_list:
        noname_indices = indices(linked_list, ["NONAME"])
        linked_list_noface = [item for item in linked_list if item != ["NOFACE"]]
        noname_indices_sim = indices(linked_list_noface, ["NONAME"])
    elif ["NONAME"] in linked_list:
        noname_indices = indices(linked_list, ["NONAME"])
        noname_indices_sim = indices(linked_list, ["NONAME"])
    else:
        noname_indices = []
        noname_indices_sim = []
    return noname_indices, noname_indices_sim, noface_indices


def cal_acc_noname(noname_dict):
    all_count = 0
    correct_count = 0
    for key in noname_dict.keys():
        sample = noname_dict[key]
        input_names = [links[0] for links in sample["gt_link_full"]]
        aligned_names = []
        for i in range(len(sample["sim_face_name_full"])):
            aligned_idx = sample["sim_face_name_full"][i].index(max(sample["sim_face_name_full"][i]))
            aligned_names.append(input_names[aligned_idx])
        
        gt_names = [links[1] for links in sample["gt_link_full"]]
        # print([name for name in gt_names if name != ["NOFACE"]])
        noname_idx = sample["noname_sim_idx"]
        # print(noname_idx, aligned_names, gt_names)
        gt_nonames = [gt_names[i] for i in noname_idx]
        aligned_nonames = [aligned_names[i] for i in noname_idx]

        total_count, acc_count = compare_gt_pred_nonames(gt_nonames, aligned_nonames)
        all_count += total_count
        correct_count += acc_count
    return all_count, correct_count


def compare_gt_pred_nonames(gt_nonames, aligned_nonames):
    total_count = len(gt_nonames)
    acc_count = 0
    for i in range(total_count):
        if aligned_nonames[i] == gt_nonames[i]:
            acc_count += 1
        else:
            continue
    return total_count, acc_count


def gen_subdict_noface(result_dict):
    out_dict = {}
    for key in result_dict.keys():
        _, _, noface_indices = get_noname_indices(result_dict[key]["gt_link"])
        if noface_indices != []:
            if len(result_dict[key]["gt_link"]) != len(result_dict[key]["sim_face_name"]) + len(noface_indices):
                print(f"check key1:{key}")
                # print(result_dict[key]["gt_link"])
                # print(result_dict[key]["sim_face_name"])
            elif key not in out_dict.keys():
                out_dict[key] = {}
                out_dict[key]["noface_sim_idx"] = noface_indices
                out_dict[key]["gt_link_full"] = result_dict[key]["gt_link"]
                out_dict[key]["sim_face_name_full"] = result_dict[key]["sim_face_name"]
            else:
                print(f"check key:{key}")
        else:
            continue
    return out_dict


def cal_acc_noface(noface_dict):
    all_count = 0
    correct_count = 0
    for key in noface_dict.keys():
        sample = noface_dict[key]
        
        aligned_indices = []
        for i in range(len(sample["sim_face_name_full"])):
            aligned_idx = sample["sim_face_name_full"][i].index(max(sample["sim_face_name_full"][i]))
            aligned_indices.append(aligned_idx)
        # gt_names = [links[1] for links in sample["gt_link_full"]]
        noface_idx = sample["noface_sim_idx"]
        # print(input_names, gt_names, aligned_indices, noface_idx)
        for i in range(len(noface_idx)):
            if noface_idx[i] in aligned_indices:
                continue
            else:
                correct_count += 1
        all_count += len(noface_idx)

    return all_count, correct_count



def gen_subdict_noname_celeb(result_dict):
    out_dict = {}
    for key in result_dict.keys():
        noname_indices, noname_indices_sim, _ = get_noname_indices(result_dict[key]["gt_link"])
        # print(noname_indices, noname_indices_sim)
        if noname_indices != [] and noname_indices_sim != []:
            if key not in out_dict.keys():
                out_dict[key] = {}
                out_dict[key]["noname_sim_idx"] = noname_indices_sim
                out_dict[key]["gt_link_full"] = result_dict[key]["gt_link"]
                out_dict[key]["sim_face_name_full"] = result_dict[key]["sim_face_name"]
            else:
                print(f"check key:{key}")
        else:
            continue
    return out_dict


def cal_acc_noname_celeb(noname_dict):
    all_count = 0
    correct_count = 0
    for key in noname_dict.keys():
        sample = noname_dict[key]
        input_names = [links[0] for links in sample["gt_link_full"]]
        aligned_names = []
        for i in range(len(sample["sim_face_name_full"])):
            aligned_idx = sample["sim_face_name_full"][i].index(max(sample["sim_face_name_full"][i]))
            aligned_names.append(input_names[aligned_idx])
        
        gt_names = [links[1] for links in sample["gt_link_full"]]
        # print([name for name in gt_names if name != ["NOFACE"]])
        noname_idx = sample["noname_sim_idx"]
        # print(noname_idx, aligned_names, gt_names)
        gt_nonames = [gt_names[i] for i in noname_idx]
        aligned_nonames = [aligned_names[i] for i in noname_idx]

        total_count, acc_count = compare_gt_pred_nonames(gt_nonames, aligned_nonames)
        all_count += total_count
        correct_count += acc_count
    return all_count, correct_count


if __name__ == "__main__":
    base_dir = os.path.join(args.sys_dir, args.base_dir_name)
    out_dir = args.out_dir

    if args.experiment_type == "phi":
        with open(os.path.join(out_dir, args.result_json_name)) as f:
            result_json = json.load(f)

        with open(os.path.join(base_dir, args.data_dict)) as f:
            data_dict = json.load(f)

        noname_dict = gen_subdict_noname(result_json)
        print(len(noname_dict))
        print(cal_acc_noname(noname_dict))

        noface_dict = gen_subdict_noface(result_json)
        print(len(noface_dict))
        print(cal_acc_noface(noface_dict))

        print(cal_f1_json_noface_noname_noneg_delete_noname_noface(data_dict, result_json, noname_dict, noface_dict, True))

        for key in result_json.keys():
            linked_list = [links[1][0] for links in result_json[key]["gt_link"]]
            for ner in linked_list:
                if ner.startswith("WRONG"):
                    print([links[0][0] for links in result_json[key]["gt_link"]])
                    print(linked_list)
                else:
                    continue
    else:
        with open(os.path.join(out_dir, args.result_json_name)) as f:
            result_json = json.load(f)
        noname_dict = gen_subdict_noname_celeb(result_json)
        print(len(noname_dict))
        print(cal_acc_noname_celeb(noname_dict))

    # counter = 0
    # for key in result_json.keys():
    #     linked_list = [links[1] for links in result_json[key]["gt_link"]]
    #     if ["NOFACE"] in linked_list and linked_list.index(["NOFACE"])>0:
    #         print(linked_list)
    #         print(linked_list.index(["NOFACE"]))
    #         counter +=1
    #     else:
    #         continue
    # print(counter)

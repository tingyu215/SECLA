import json
import os
import argparse
from tqdm import tqdm
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument("--sys_dir", type=str, default="/cw/liir_code/NoCsBack/tingyu/FaceNaming")
parser.add_argument("--base_dir_name", type=str, default="CelebrityTo")
parser.add_argument("--out_dir", type=str, default="/export/home2/NoCsBack/working/tingyu/face_naming/celeb")
parser.add_argument("--result_json_name", type=str, default="unsup_frag_lname_two5-proj_dim:128_biasTrue_1.0data:train_loss:batch-0.15-agree-normal-diag_bsz:20_shuffle-True_epoch5_op:adam_lr0.0003_nonameTrue_True_textModelbert-uncased_finetune-False_mean-True-True-layerS-4.pt.json")
parser.add_argument("--data_type", type=str, default="celeb")
parser.add_argument("--dict_name", type=str, default="celeb_dict_2name_allname.json")
parser.add_argument("--add_noname", type=bool, default=True)

args = parser.parse_args()


def flatten_list(lst):
    return [a for k in lst for a in k]


def count_name_occur(data_dict):
    name_list = []
    for key in tqdm(data_dict.keys()):
        for name in data_dict[key]["name_list"]:
            name_list.append(name)
    return Counter(name_list)


def make_unique_name_dict(data_dict):
    name_counts = count_name_occur(data_dict)

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


def count_by_name(align_dict_dir):
    """
    calculate accuracy by each name
    """
    name_count_dict = {}

    with open(align_dict_dir) as f:
        align_dict = json.load(f)
    
    for key, sample in tqdm(align_dict.items()):
        name_list = flatten_list(sample["name_list"])
        pred_list = sample["pred_list"]
        for i in range(len(pred_list)):
            gt_name = name_list[i]
            if gt_name in name_count_dict.keys():
                if gt_name == pred_list[i]:
                    name_count_dict[gt_name]["acc_count"] += 1
                    name_count_dict[gt_name]["all_count"] += 1
                    if len(name_count_dict[gt_name]["img_dir"]) == 0:
                        name_count_dict[gt_name]["img_dir"] = [key.replace("/cw/working-rose/CelebTo/images_ct/", "")]
                    else:
                        name_count_dict[gt_name]["img_dir"].append(key.replace("/cw/working-rose/CelebTo/images_ct/", ""))
                else:
                    name_count_dict[gt_name]["all_count"] += 1
                    if len(name_count_dict[gt_name]["img_dir"]) == 0:
                        name_count_dict[gt_name]["img_dir"] = [key.replace("/cw/working-rose/CelebTo/images_ct/", "")]
                    else:
                        name_count_dict[gt_name]["img_dir"].append(key.replace("/cw/working-rose/CelebTo/images_ct/", ""))
            else:
                if gt_name == pred_list[i]:
                    name_count_dict[gt_name] = {}
                    name_count_dict[gt_name]["acc_count"] = 1
                    name_count_dict[gt_name]["all_count"] = 1
                    if "img_dir" not in name_count_dict[gt_name].keys():
                        name_count_dict[gt_name]["img_dir"] = [key.replace("/cw/working-rose/CelebTo/images_ct/", "")]
                    else:
                        name_count_dict[gt_name]["img_dir"].append(key.replace("/cw/working-rose/CelebTo/images_ct/", ""))
                else:
                    name_count_dict[gt_name] = {}
                    name_count_dict[gt_name]["acc_count"] = 0
                    name_count_dict[gt_name]["all_count"] = 1
                    if "img_dir" not in name_count_dict[gt_name].keys():
                        name_count_dict[gt_name]["img_dir"] = [key.replace("/cw/working-rose/CelebTo/images_ct/", "")]
                    else:
                        name_count_dict[gt_name]["img_dir"].append(key.replace("/cw/working-rose/CelebTo/images_ct/", ""))
    return name_count_dict


def acc_by_name(name_count_dict):
    name_acc_dict = {}
    name_acc_dict["1~15"] = {}
    name_acc_dict["15~30"] = {}
    name_acc_dict["30~50"] = {}
    name_acc_dict["50~100"] = {}
    name_acc_dict["100~200"] = {}
    name_acc_dict[">200"] = {}
    for name, sample in tqdm(name_count_dict.items()):
        all_count = sample["all_count"]
        if all_count < 15:
            name_acc_dict["1~15"][name] = sample["acc_count"] / all_count
        elif all_count < 30:
            name_acc_dict["15~30"][name] = sample["acc_count"] / all_count 
        elif all_count < 50:
            name_acc_dict["30~50"][name] = sample["acc_count"] / all_count    
        elif all_count < 100:
            name_acc_dict["50~100"][name] = sample["acc_count"] / all_count 
        elif all_count < 200:
            name_acc_dict["100~200"][name] = sample["acc_count"] / all_count 
        else:
            name_acc_dict[">200"][name] = sample["acc_count"] / all_count 
    return name_acc_dict


def cal_acc_by_group(name_count_dict):
    name_acc_dict = acc_by_name(name_count_dict)
    for group in name_acc_dict.keys():
        acc_list = [acc for acc in name_acc_dict[group].values()]
        if len(acc_list) == 0:
            print("{} group has no sample".format(group))
        else:
            print("{} group has {} sample, average acc={}".format(group, len(acc_list), sum(acc_list) / len(acc_list)))
    return name_acc_dict


def cal_dist_by_name(name_count_dict):
    name_count_dict_few = {}
    acc_list_few = []
    acc_list_more = []
    for key, sample in name_count_dict.items():
        if sample["all_count"] <= 10:
            name_count_dict_few[key] = {}
            name_count_dict_few[key] = sample
            acc_list_few.append(sample["acc_count"] / sample["all_count"])
        else:
            acc_list_more.append(sample["acc_count"] / sample["all_count"])
    avg_acc_few = sum(acc_list_few) / len(acc_list_few)
    avg_acc_more = sum(acc_list_more) / len(acc_list_more)
    return name_count_dict_few, avg_acc_few, avg_acc_more


def make_dict_without_few(name_count_dict_few, data_dict):
    del_img_dirs = []
    for sample in name_count_dict_few.values():
        del_img_dirs.extend(sample["img_dir"])
    print(f"number of deleted samples:{len(del_img_dirs)}")
    
    data_dict_nofew = {}
    for key, sample in data_dict.items():
        if sample["img_dir"] in del_img_dirs:
            continue
        else:
            data_dict_nofew[key] = {}
            data_dict_nofew[key] = sample
    return data_dict_nofew


if __name__ == "__main__":
    # print(args.add_noname)
    # dict_dir = os.path.join(args.sys_dir, args.base_dir_name)
    # out_dict_dir = args.out_dir
    # align_dict_dir = os.path.join(out_dict_dir, "align"+args.result_json_name)


    # name_count_dict = count_by_name(align_dict_dir)
    # print(len(name_count_dict))
    # print(name_count_dict["Hayley Marie Norman"])

    # name_count_dict_few, avg_acc_few, avg_acc_more = cal_dist_by_name(name_count_dict)
    # print(len(name_count_dict_few))
    # print(avg_acc_few)
    # print(avg_acc_more)

    # with open(os.path.join(dict_dir, args.dict_name)) as f:
    #     data_dict = json.load(f)

    # data_dict_nofew = make_dict_without_few(name_count_dict_few, data_dict)
    # print(len(data_dict))
    # print(len(data_dict_nofew))

    # unique_name_nofew = make_unique_name_dict(data_dict_nofew)
    # for key,sample in unique_name_nofew.items():
    #     if len(sample["bbox"]) < 10:
    #         print(key)
    

    # with open(os.path.join(dict_dir, "celeb_dict_2name_allname_15more.json"), "w") as f:
    #     json.dump(data_dict_nofew, f)

    # test_dict_dir = os.path.join(out_dict_dir, "alignunsup_frag_5more_two5-proj_dim:128_biasTrue_1.0data:train_loss:batch-0.15-agree-normal-diag_bsz:20_shuffle-True_epoch5_op:adam_lr0.0003_nonameTrue_True_textModelbert-uncased_finetune-False_mean-True-True-layerS-4.pt.json")
    # with open(test_dict_dir) as f:
    #     test_dict = json.load(f)
    # count = 0
    # for sample in test_dict.values():
    #     if "NONAME" in sample['pred_list']:
    #         count += 1
    # print(count)

    # with open(os.path.join(dict_dir, "celeb_dict.json")) as f:
    #     data_dict_full = json.load(f)

    # data_dict_5less = {}
    # data_dict_5more = {}
    # for key, sample in data_dict_full.items():
    #     if len(sample["face_id"]) <= 5:
    #         data_dict_5less[key] = {}
    #         data_dict_5less[key] = sample
    #     else:
    #         data_dict_5more[key] = {}
    #         data_dict_5more[key] = sample

    # unique_name_5more = count_name_occur(data_dict_5more)
    # unique_name_5less = count_name_occur(data_dict_5less)
    # print(len(unique_name_5more))
    # print(len(unique_name_5less))



    # test_dict_count = count_by_name(test_dict_dir)
    # for key, sample in test_dict_count.items():
    #     if sample["all_count"]<=15:
    #         print(key)

    # with open(os.path.join(dict_dir, "name_count_dict_3epoch.json"), "w") as f:
    #     json.dump(name_count_dict, f)
    
    # name_acc_dict = cal_acc_by_group(name_count_dict)
    # with open(os.path.join(dict_dir, "name_acc_dict_3epoch.json"), "w") as f:
    #     json.dump(name_acc_dict, f)



    # with open("/cw/liir_code/NoCsBack/tingyu/FaceNaming/Berg/gt_dict_cleaned_phi_face_name_one_unique_middle_pdist.json") as f:
    #     data_dict_unique_pdist_phi = json.load(f)
    
    # cleaned_dict_pdist_phi = {}
    # for key, sample in data_dict_unique_pdist_phi.items():
    #     if type(sample["face_x"][0]) is list:
    #         cleaned_dict_pdist_phi[key] = {}
    #         cleaned_dict_pdist_phi[key] = sample
    #     else:
    #         cleaned_dict_pdist_phi[key] = {}
    #         cleaned_dict_pdist_phi[key]["img_name"] = [sample["img_name"]]
    #         cleaned_dict_pdist_phi[key]["face_x"] = [sample["face_x"]]
    #         cleaned_dict_pdist_phi[key]["face_y"] = [sample["face_y"]]
    #         cleaned_dict_pdist_phi[key]["face_size"] = [sample["face_size"]]
        
    # with open("/cw/liir_code/NoCsBack/tingyu/FaceNaming/Berg/gt_dict_cleaned_phi_face_name_one_unique_middle_pdist_new.json", "w") as f:
    #     json.dump(cleaned_dict_pdist_phi, f)
    

    # with open("/cw/liir_code/NoCsBack/tingyu/FaceNaming/Berg/gt_dict_cleaned_phi_face_name_one_unique_middle_sim.json") as f:
    #     data_dict_unique_sim_phi = json.load(f)
    
    # cleaned_dict_sim_phi = {}
    # for key, sample in data_dict_unique_sim_phi.items():
    #     if type(sample["face_x"][0]) is list:
    #         cleaned_dict_sim_phi[key] = {}
    #         cleaned_dict_sim_phi[key] = sample
    #     else:
    #         cleaned_dict_sim_phi[key] = {}
    #         cleaned_dict_sim_phi[key]["img_name"] = [sample["img_name"]]
    #         cleaned_dict_sim_phi[key]["face_x"] = [sample["face_x"]]
    #         cleaned_dict_sim_phi[key]["face_y"] = [sample["face_y"]]
    #         cleaned_dict_sim_phi[key]["face_size"] = [sample["face_size"]]
        
    # with open("/cw/liir_code/NoCsBack/tingyu/FaceNaming/Berg/gt_dict_cleaned_phi_face_name_one_unique_middle_sim_new.json", "w") as f:
    #     json.dump(cleaned_dict_sim_phi, f)

    # with open("/cw/liir_code/NoCsBack/tingyu/FaceNaming/CelebrityTo/celeb_allname_unique_middle_pdist.json") as f:
    #     data_dict_unique_pdist = json.load(f)
    
    # cleaned_dict_pdist = {}
    # for key, sample in data_dict_unique_pdist.items():
    #     if type(sample["img_dir"]) is list:
    #         cleaned_dict_pdist[key] = {}
    #         cleaned_dict_pdist[key] = sample
    #     else:
    #         cleaned_dict_pdist[key] = {}
    #         cleaned_dict_pdist[key]["img_dir"] = [sample["img_dir"]]
    #         cleaned_dict_pdist[key]["bbox"] = [sample["bbox"]]
        
    # with open("/cw/liir_code/NoCsBack/tingyu/FaceNaming/CelebrityTo/celeb_allname_unique_middle_pdist_new.json", "w") as f:
    #     json.dump(cleaned_dict_pdist, f)
    

    # with open("/cw/liir_code/NoCsBack/tingyu/FaceNaming/CelebrityTo/celeb_allname_unique_middle_sim.json") as f:
    #     data_dict_unique_sim = json.load(f)
    
    # cleaned_dict_sim = {}
    # for key, sample in data_dict_unique_sim.items():
    #     if type(sample["img_dir"]) is list:
    #         cleaned_dict_sim[key] = {}
    #         cleaned_dict_sim[key] = sample
    #     else:
    #         cleaned_dict_sim[key] = {}
    #         cleaned_dict_sim[key]["img_dir"] = [sample["img_dir"]]
    #         cleaned_dict_sim[key]["bbox"] = [sample["bbox"]]
        
    # with open("/cw/liir_code/NoCsBack/tingyu/FaceNaming/CelebrityTo/celeb_allname_unique_middle_sim_new.json", "w") as f:
    #     json.dump(cleaned_dict_sim, f)

    with open("/cw/liir_code/NoCsBack/tingyu/FaceNaming/Berg/gt_dict_cleaned_phi_face_name_2face.json") as f:
        dict_phi_2face = json.load(f)

    nonamewrong_counter = 0
    nofacewrong_counter = 0
    all_counter = 0
    unique_name_list = []
    for sample in dict_phi_2face.values():
        for ner in sample["ner"]:
            all_counter += 1
            if ner == "NONAMEWRONG":
                nonamewrong_counter += 1
            elif ner == "NOFACEWRONG":
                nofacewrong_counter += 1
            else:
                continue
    
    for sample in dict_phi_2face.values():
        for ner in sample["name_list"]:
            if ner not in unique_name_list:
                unique_name_list.append(ner)
            

    print(f"{nonamewrong_counter} NONAMEWRONG")
    print(f"{nofacewrong_counter} NOFACEWRONG")
    print(f"{all_counter} Links")
    print(len(dict_phi_2face))
    print(len(unique_name_list))


    with open("/cw/liir_code/NoCsBack/tingyu/FaceNaming/CelebrityTo/celeb_dict_2name_unique_allname.json") as f:
        celeb_2name_unique_dict = json.load(f)
    print(len(celeb_2name_unique_dict))
    print(len(celeb_2name_unique_dict["Abigail Breslin"]["img_dir"]))


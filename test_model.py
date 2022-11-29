from email.policy import default
import json
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import re
import argparse

from scripts.dataset import CelebDataset, FaceDataset

from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
from transformers import BertTokenizer, BertModel

parser = argparse.ArgumentParser()
parser.add_argument("--sys_dir", type=str, default="/FaceNaming")
parser.add_argument("--experiment_type", type=str, default="unsup_frag")
parser.add_argument("--base_dir_name", type=str, default="Berg")
parser.add_argument("--base_dir", type=str, default="/CelebTo/images_ct")
parser.add_argument("--dict_name", type=str, default="gt_dict_cleaned.json")
parser.add_argument("--gpu_ids", type=str, default="1")

parser.add_argument("--waldo_dir", type=str, default="/Waldo")
parser.add_argument("--waldo_model_name", type=str, default="waldo-unsup_frag_two5-proj_dim:128_biasTrue_data:train_loss:batch-0.25-agree-normal-full_bsz:20_epoch3_op:adam_lr0.0003_nonameTrue_True_textModelbert-uncased_finetune-False_mean-True-False-layerS-4.pt")

parser.add_argument("--alpha", type=float, default=0.15)
parser.add_argument("--agree_type", type=str, default="full")
parser.add_argument("--data_name", type=str, default="allname")

parser.add_argument("--add_extra_proj", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--beta_incre", type=float, default=0.5)

parser.add_argument("--data_type", type=str, default="all")
parser.add_argument("--data_dict", type=str, default="gt_dict_cleaned_phi_face_name.json")

parser.add_argument("--text_model_type", type=str, default="bert-uncased")
parser.add_argument("--charbert_dir", type=str, default="/FaceNaming/models/character_bert/pretrained-models/general_character_bert")
parser.add_argument("--text_model", type=str, default="bert-base-uncased")
parser.add_argument("--face_model", type=str, default="vggface2")
parser.add_argument("--special_token_list", type=list, default=["[Ns]", "[Ne]"])
parser.add_argument("--test_batch_size", type=int, default=1)

parser.add_argument("--proj_type", type=str, default="two5")
parser.add_argument("--fine_tune", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--use_mean", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--layer_start", type=int, default=-4)
parser.add_argument("--layer_end", default=None)
parser.add_argument("--add_special_token", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--use_name_ner", type=str, default="ner")
parser.add_argument("--add_noname", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--cons_noname", default=False, type=lambda x: (str(x).lower() == 'true'))

args = parser.parse_args()


class UnsupIncre(nn.Module):
    def __init__(self, model_one_stage, beta_incre, ner_dim, freeze_stage1=True, proj_type="two5", add_bias=True, n_features=512, proj_dim=512):
        super(UnsupIncre, self).__init__()

        if freeze_stage1:
            for param in model_one_stage.parameters():
                param.requires_grad = False
            self.model_one_stage = model_one_stage
        else:
            self.model_one_stage = model_one_stage

        self.beta_incre = beta_incre

        self.ner_dim = ner_dim
        self.proj_type = proj_type  # proj_type=one: one projector for both; two: separate projector
        self.n_features = n_features

        if self.proj_type == "two5":
            self.projector = nn.Sequential(
                nn.Linear(self.n_features, self.n_features, bias=False),
                nn.ReLU(),
                nn.Linear(self.n_features, proj_dim, bias=add_bias),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim, bias=add_bias),
            )

            self.ner_projector = nn.Sequential(
                nn.Linear(self.n_features, self.n_features, bias=False),
                nn.ReLU(),
                nn.Linear(self.n_features, proj_dim, bias=add_bias),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim, bias=add_bias),
            )
        elif self.proj_type == "two9":
            self.projector = nn.Sequential(
                nn.Linear(self.n_features, self.n_features, bias=False),
                nn.ReLU(),
                nn.Linear(self.n_features, self.n_features // 2, bias=add_bias),
                nn.ReLU(),
                nn.Linear(self.n_features // 2, self.n_features // 2, bias=add_bias),
                nn.ReLU(),
                nn.Linear(self.n_features // 2, proj_dim, bias=add_bias),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim, bias=add_bias),
            )
            self.ner_projector = nn.Sequential(
                nn.Linear(self.n_features, self.n_features, bias=False),
                nn.ReLU(),
                nn.Linear(self.n_features, self.n_features // 2, bias=add_bias),
                nn.ReLU(),
                nn.Linear(self.n_features // 2, self.n_features // 2, bias=add_bias),
                nn.ReLU(),
                nn.Linear(self.n_features // 2, proj_dim, bias=add_bias),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim, bias=add_bias),
            )
        elif self.proj_type == "two9_drop":
            self.projector = nn.Sequential(
                nn.Linear(self.n_features, self.n_features, bias=False),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.n_features, self.n_features // 2, bias=add_bias),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.n_features // 2, self.n_features // 2, bias=add_bias),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.n_features // 2, proj_dim, bias=add_bias),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(proj_dim, proj_dim, bias=add_bias),
            )
            self.ner_projector = nn.Sequential(
                nn.Linear(self.n_features, self.n_features, bias=False),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.n_features, self.n_features // 2, bias=add_bias),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.n_features // 2, self.n_features // 2, bias=add_bias),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.n_features // 2, proj_dim, bias=add_bias),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(proj_dim, proj_dim, bias=add_bias),
            )
        elif self.proj_type == "two4":
            self.projector = nn.Sequential(
                nn.Linear(self.n_features, self.n_features, bias=False),
                nn.ReLU(),
                nn.Linear(self.n_features, proj_dim, bias=add_bias),
                nn.ReLU(),
            )
            self.ner_projector = nn.Sequential(
                nn.Linear(self.n_features, self.n_features, bias=False),
                nn.ReLU(),
                nn.Linear(self.n_features, proj_dim, bias=add_bias),
                nn.ReLU(),
            )
        elif self.proj_type == "two4_drop":
            self.projector = nn.Sequential(
                nn.Linear(self.n_features, self.n_features, bias=False),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(self.n_features, proj_dim, bias=add_bias),
                nn.Dropout(0.3),
                nn.ReLU(),
            )
            self.ner_projector = nn.Sequential(
                nn.Linear(self.n_features, self.n_features, bias=False),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(self.n_features, proj_dim, bias=add_bias),
                nn.Dropout(0.3),
                nn.ReLU(),
            )
        else:
            self.projector = nn.Sequential(
                nn.Linear(self.n_features, proj_dim, bias=add_bias),
                nn.Dropout(0.3),
                nn.ReLU(),
            )

            self.ner_projector = nn.Sequential(
                nn.Linear(self.n_features, proj_dim, bias=add_bias),
                nn.Dropout(0.3),
                nn.ReLU(),
            )
        self.ner_proj = nn.Sequential(
            nn.Linear(self.ner_dim, self.n_features, bias=False),
            # nn.Linear(self.n_features, self.n_features, bias=add_bias),
            nn.ReLU(),
        )

    def forward(self, enc_face_emb, enc_ner_emb):
        if self.proj_type == "no_face":  # no projector for face features, this case proj_dim=512
            face_z_i = enc_face_emb
        else:
            face_z_i = (1-self.beta_incre) * self.projector(enc_face_emb) \
                       + self.beta_incre * self.model_one_stage.projector(enc_face_emb)

        ner_z_j = (1-self.beta_incre) * self.ner_proj(enc_ner_emb) \
                  + self.beta_incre * self.model_one_stage.ner_proj(enc_ner_emb)  # size 1*1*768 --> 1*1*512
        if self.proj_type == "one":
            ner_z_j = (1-self.beta_incre) * self.projector(ner_z_j) \
                      + self.beta_incre * self.model_one_stage.projector(ner_z_j)
        else:
            ner_z_j = (1-self.beta_incre) * self.ner_projector(ner_z_j) \
                      + self.beta_incre * self.model_one_stage.ner_projector(ner_z_j)

        return face_z_i, ner_z_j


def test_waldo(model, test_loader, DEVICE):
    unsup_align_out = {}
    with torch.no_grad():

        for idx, data in tqdm(enumerate(test_loader)):
            image_name, all_faces, ner_pos_i, ner_list, gt_ner, gt_link, names, ner_ids = data["image_name"][0], data["face_emb"], data["ner_features"], data["ner_list"], data["gt_ner"], data["gt_link"], data["names"], data["ner_ids"]

            num_face_i = all_faces.size()[2]
            face_list_all = []
            for j in range(num_face_i):
                if args.add_extra_proj:

                    face_z_i = (1 - args.beta_incre) * model.projector(all_faces.squeeze(0).squeeze(0)[j].cuda()) \
                               + args.beta_incre * model.model_one_stage.projector(all_faces.squeeze(0).squeeze(0)[j].cuda())

                    if face_z_i.dim() < 1:
                        face_z_i = face_z_i.unsqueeze(0)

                    ner_i = ner_pos_i

                    ner_z_j = (1 - args.beta_incre) * model.ner_proj(ner_i.squeeze(0).squeeze(0).to(DEVICE)) + args.beta_incre * model.model_one_stage.ner_proj(ner_i.squeeze(0).squeeze(0).to(DEVICE))

                    if args.proj_type == "one":
                        ner_z_all = (1 - args.beta_incre) * model.projector(ner_z_j) \
                                    + args.beta_incre * model.model_one_stage.projector(ner_z_j)
                    else:
                        ner_z_all = (1 - args.beta_incre) * model.ner_projector(ner_z_j) \
                                    + args.beta_incre * model.model_one_stage.ner_projector(ner_z_j)
                else:
                    face_z_i = model.projector(all_faces.squeeze(0).squeeze(0)[j].cuda())

                    if face_z_i.dim() < 1:
                        face_z_i = face_z_i.unsqueeze(0)

                    ner_i = ner_pos_i

                    if args.proj_type == "one":
                        ner_z_all = model.projector(model.ner_proj(ner_i.squeeze(0).squeeze(0).to(DEVICE)))
                    elif args.fine_tune:
                        enc_ner_emb = model.create_ner_emb(ner_ids)
                        ner_z_all = model.ner_projector(model.ner_proj(enc_ner_emb.squeeze(0).to(DEVICE)))
                    else:
                        ner_z_all = model.ner_projector(model.ner_proj(ner_i.squeeze(0).squeeze(0).to(DEVICE)))

                sim_all = torch.matmul(face_z_i, torch.transpose(ner_z_all, 0, 1))
                face_list_all.append(sim_all.tolist())

            unsup_align_out[image_name] = {}
            if args.use_name_ner == "ner":
                unsup_align_out[image_name]["ner_list"] = ner_list
                unsup_align_out[image_name]["sim_face_name"] = face_list_all
                unsup_align_out[image_name]["gt_ner"] = gt_ner

            else:
                unsup_align_out[image_name]["name_list"] = names
                unsup_align_out[image_name]["gt_link"] = gt_link
                unsup_align_out[image_name]["sim_face_name"] = face_list_all

    return unsup_align_out



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

    with open(os.path.join(out_dir, out_file_name), "w") as f:
        json.dump(align_result_dict, f)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


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

    with open(os.path.join(out_dir, out_file_name), "w") as f:
        json.dump(align_result_dict, f)

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
    print(args.waldo_model_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args.data_dict)

    sys_dir = args.sys_dir
    if args.experiment_type == "celeb":
        base_dir = args.base_dir
    else:
        base_dir = os.path.join(sys_dir, args.base_dir_name)

    tokenizer = BertTokenizer.from_pretrained(args.text_model)
    facenet = InceptionResnetV1(pretrained=args.face_model).eval()
    if args.text_model_type == "bert-uncased" or args.text_model_type == "bert-cased" or args.text_model_type == "ernie":
        text_model = BertModel.from_pretrained(args.text_model, output_hidden_states=True)
    else:
        text_model = {}
    indexer = {}
    special_token_dict = {"additional_special_tokens": args.special_token_list}

    if args.experiment_type == "celeb" or args.experiment_type == "celeb_noneg":
        face_data = CelebDataset(base_dir,
                            tokenizer,
                            indexer,
                            special_token_dict,
                            "cpu",
                            facenet,
                            text_model,
                            text_model_type=args.text_model_type,
                            use_mean=args.use_mean,
                            layer_start=args.layer_start,
                            layer_end=args.layer_end,
                            add_special_token=args.add_special_token,
                            use_name_ner=args.use_name_ner,
                            add_noname=args.add_noname,
                            cons_noname=args.cons_noname,
                            dict_name=args.dict_name)
    else:
        face_data = FaceDataset(base_dir,
                                tokenizer,
                                indexer,
                                special_token_dict,
                                "cpu",
                                facenet,
                                text_model,
                                text_model_type=args.text_model_type,
                                use_mean=args.use_mean,
                                layer_start=args.layer_start,
                                layer_end=args.layer_end,
                                add_special_token=args.add_special_token,
                                use_name_ner=args.use_name_ner,
                                add_noname=args.add_noname,
                                cons_noname=args.cons_noname,
                                dict_name=args.dict_name)
    all_loader_test = DataLoader(face_data, batch_size=args.test_batch_size, num_workers=4)

    unsup_frag_net = torch.load(os.path.join(args.waldo_dir, args.waldo_model_name))

    with open(os.path.join(args.waldo_dir, args.waldo_model_name[:-3] + ".json")) as f:
        results_dict = json.load(f)

    with open(os.path.join(base_dir, args.data_dict)) as f:
        data_dict = json.load(f)

    if args.experiment_type == "celeb":
        out_file_name = "align" + args.waldo_model_name[:-3] + ".json"
        print(cal_f1_json_celeb(results_dict, args.waldo_dir, out_file_name))
    elif args.experiment_type == "celeb_noneg":
        out_file_name = "align" + args.waldo_model_name[:-3] + "noneg.json"
        print(cal_f1_json_celeb_noneg(results_dict, args.waldo_dir, out_file_name))
    else:
        print(cal_f1_json_noface_noname(data_dict, results_dict, args.add_noname))


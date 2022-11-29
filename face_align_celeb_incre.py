from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import T
import time

from scripts.dataset import CelebDataset, crop_resize

import os
from tqdm import tqdm
import numpy as np
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, RandomSampler
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd
from facenet_pytorch import InceptionResnetV1
import torch.distributed as dist
from torch.optim import SGD
from transformers import BertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

from models.face_align_model import UnsupFragAlign, UnsupFragAlign_FineTune, FragAlignLoss, GlobalRankLoss, BatchSoftmax, BatchSoftmaxSplit, batch_softmax, batch_agreement
from models.face_align_model_incre import UnsupIncre
from modules.lars import LARS

# from models.character_bert.modeling.character_bert import CharacterBertModel
# from models.character_bert.utils.character_cnn import CharacterIndexer

from PIL import Image
from torchvision import transforms
from collections import Counter
import logging
import datetime


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--base_dir", type=str, default="/cw/working-rose/CelebTo/images_ct")
parser.add_argument("--dict_dir", type=str, default="/cw/liir_code/NoCsBack/tingyu/FaceNaming/CelebTo")
parser.add_argument("--out_dir", type=str, default="/cw/working-rose/tingyu/FaceNaming/results/celeb")
parser.add_argument("--full_dict_name", type=str, default="celeb_dict.json")
parser.add_argument("--2name_dict_name" ,type=str, default="celeb_dict_2name.json")
parser.add_argument("--dict_name", type=str, default="celeb_dict_2name_rest.json")
parser.add_argument("--data_name", type=str, default="2name")

parser.add_argument("--add_extra_proj", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--freeze_stage1", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--manual_add_one", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--split_sum", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--nomax", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--add_sample", type=str, default="no")
parser.add_argument("--add_to" ,type=str, default="match")
parser.add_argument("--make_new", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--combined_direction", type=str, default="both")
parser.add_argument("--face_contras", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--face_con_direction", type=str, default="both")
parser.add_argument("--only_face_con", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--face_con_replace_diag", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--add_d_one", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--match_proto_agree", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--noname_to_match", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--add_nullface", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--no_stage1", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--unique_name_avg_face_dict_name", type=str, default="celeb_dict_2name_avg_face.json")

parser.add_argument("--sample_type", type=str, default="True")
parser.add_argument("--unique_name_dict_name", type=str, default="celeb_dict_2name_unique.json")

parser.add_argument("--batch_rest", type=int, default=1)
parser.add_argument("--stage1_model_dir", type=str, default="/export/home1/NoCsBack/working/tingyu/face_naming/celeb")
parser.add_argument("--stage1_model_name", type=str, default="unsup_frag_e_one_two5-proj_dim:128_biasTrue_1.0data:train_loss:batch-0.25-agree-normal-full_bsz:20_shuffle-True_epoch15_op:adam_lr0.0003_nonameTrue_True_textModelbert-uncased_finetune-False_mean-True-True-layerS-4.pt")
parser.add_argument("--beta_incre", type=float, default=0.5)

parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--gpu_ids", type=str, default="1")
parser.add_argument("--num_gpu", type=int, default=1)
parser.add_argument("--seed", type=int, default=684331)
parser.add_argument("--fine_tune", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--loss_type", type=str, default="all")
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--direction", type=str, default="one")
parser.add_argument("--pos_factor", type=float, default=1)
parser.add_argument("--neg_factor", type=float, default=1)
parser.add_argument("--delta", type=float, default=0.1)
parser.add_argument("--smooth_term", type=float, default=3)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--optimizer_type", type=str, default="lars")

parser.add_argument("--data_type", type=str, default="test")
parser.add_argument("--add_context", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--special_token_list", type=list, default=["[Ns]", "[Ne]"])
parser.add_argument("--data_percent", type=float, default=1)
parser.add_argument("--proj_type", type=str, default="one")
parser.add_argument("--proj_dim", type=int, default=512)
parser.add_argument("--n_features", type=int, default=512)
parser.add_argument("--ner_dim", type=int, default=768)
parser.add_argument("--use_name_ner", type=str, default="ner")
parser.add_argument("--add_noname", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--cons_noname", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--add_bias", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--model_name", type=str, default="unsup_frag")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr_scheduler_tmax", type=int, default=32)
parser.add_argument("--sgd_momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--adam_beta1", type=float, default=0.9)
parser.add_argument("--adam_beta2", type=float, default=0.999)
parser.add_argument("--adam_epsilon", type=float, default=1e-6)

parser.add_argument("--num_epoch", type=int, default=12)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--val_batch_size", type=int, default=16)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument("--shuffle", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--text_model_type", type=str, default="bert-uncased")
parser.add_argument("--charbert_dir", type=str, default="/cw/working-rose/tingyu/FaceNaming/models/character_bert/pretrained-models/general_character_bert")
parser.add_argument("--text_model", type=str, default="bert-base-uncased")
parser.add_argument("--face_model", type=str, default="vggface2")

parser.add_argument("--use_mean", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--layer_start", type=int, default=-4)
parser.add_argument("--layer_end", default=None)
parser.add_argument("--add_special_token", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--margin", type=float, default=0.2)
parser.add_argument("--agree_type", type=str, default="full")
parser.add_argument("--max_type", type=str, default="normal")
parser.add_argument("--use_onehot", default=False, type=lambda x: (str(x).lower() == 'true'))

args = parser.parse_args()


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0:logging.DEBUG, 1:logging.INFO, 2:logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def combined_frag_loss(face_j_match, face_j_add, ner_j_match, direction):
    if direction == "face":
        face_ner_match = torch.matmul(ner_j_match.unsqueeze(1), face_j_match.permute(0, 2, 1))
        face_ner_match_add = torch.matmul(ner_j_match.unsqueeze(1), face_j_add.permute(0, 2, 1))
        # replace diagonal elements with matched faces
        face_ner_match_add.diagonal().copy_(face_ner_match.diagonal())

        loss = batch_softmax(face_ner_match_add, args.max_type)
    elif direction == "name":
        ner_face_match = torch.matmul(face_j_match.unsqueeze(1), ner_j_match.permute(0, 2, 1))
        ner_face_match_add = torch.matmul(face_j_add.unsqueeze(1), ner_j_match.permute(0, 2, 1))
        # replace diagonal elements with matched faces
        ner_face_match_add.diagonal().copy_(ner_face_match.diagonal())

        loss = batch_softmax(ner_face_match_add, args.max_type)
    elif direction == "both":
        face_ner_match = torch.matmul(ner_j_match.unsqueeze(1), face_j_match.permute(0, 2, 1))
        face_ner_match_add = torch.matmul(ner_j_match.unsqueeze(1), face_j_add.permute(0, 2, 1))
        # replace diagonal elements with matched faces
        face_ner_match_add.diagonal().copy_(face_ner_match.diagonal())
        loss1 = batch_softmax(face_ner_match_add, args.max_type)
        
        ner_face_match = torch.matmul(face_j_match.unsqueeze(1), ner_j_match.permute(0, 2, 1))
        ner_face_match_add = torch.matmul(face_j_add.unsqueeze(1), ner_j_match.permute(0, 2, 1))
        # replace diagonal elements with matched faces
        ner_face_match_add.diagonal().copy_(ner_face_match.diagonal())

        loss2 = batch_softmax(ner_face_match_add, args.max_type)

        loss = loss1 + loss2
    else:
        face_ner_match = torch.matmul(ner_j_match.unsqueeze(1), face_j_match.permute(0, 2, 1))
        face_ner_match_add = torch.matmul(ner_j_match.unsqueeze(1), face_j_add.permute(0, 2, 1))
        # replace diagonal elements with matched faces
        face_ner_match_add.diagonal().copy_(face_ner_match.diagonal())

        loss1 = batch_softmax(face_ner_match_add, args.max_type)
        
        ner_face_match = torch.matmul(face_j_match.unsqueeze(1), ner_j_match.permute(0, 2, 1))
        ner_face_match_add = torch.matmul(face_j_add.unsqueeze(1), ner_j_match.permute(0, 2, 1))
        # replace diagonal elements with matched faces
        ner_face_match_add.diagonal().copy_(ner_face_match.diagonal())

        loss2 = batch_softmax(ner_face_match_add, args.max_type)

        loss3 = batch_agreement(face_ner_match_add, ner_face_match_add, args.agree_type, args.max_type)

        loss = loss1 + loss2 + args.alpha * loss3
    return loss


def face_contras_loss(face_j_match, face_j_add, direction):
    face_match = torch.matmul(face_j_match.unsqueeze(1), face_j_match.permute(0, 2, 1))
    if direction == "face":
        if args.face_con_replace_diag:
            face_proto_match = torch.matmul(face_j_match.unsqueeze(1), face_j_add.permute(0, 2, 1))
            # replace diagonal elements with matched faces
            face_proto_match.diagonal().copy_(face_match.diagonal())
        else:
            face_proto_match = torch.matmul(face_j_match.unsqueeze(1), face_j_add.permute(0, 2, 1))
        loss = batch_softmax(face_proto_match, args.max_type)
    elif direction == "proto":
        if args.face_con_replace_diag:
            proto_face_match = torch.matmul(face_j_add.unsqueeze(1), face_j_match.permute(0, 2, 1))
            # replace diagonal elements with matched faces
            proto_face_match.diagonal().copy_(face_match.diagonal())
        else:
            proto_face_match = torch.matmul(face_j_add.unsqueeze(1), face_j_match.permute(0, 2, 1))
        loss = batch_softmax(proto_face_match, args.max_type)
    elif direction == "both":
        if args.face_con_replace_diag:
            face_proto_match = torch.matmul(face_j_match.unsqueeze(1), face_j_add.permute(0, 2, 1))
            # replace diagonal elements with matched faces
            face_proto_match.diagonal().copy_(face_match.diagonal())
        else:
            face_proto_match = torch.matmul(face_j_match.unsqueeze(1), face_j_add.permute(0, 2, 1))
        loss1 = batch_softmax(face_proto_match, args.max_type)
        
        if args.face_con_replace_diag:
            proto_face_match = torch.matmul(face_j_add.unsqueeze(1), face_j_match.permute(0, 2, 1))
            # replace diagonal elements with matched faces
            proto_face_match.diagonal().copy_(face_match.diagonal())
        else:
            proto_face_match = torch.matmul(face_j_add.unsqueeze(1), face_j_match.permute(0, 2, 1))
        loss2 = batch_softmax(proto_face_match, args.max_type)

        loss = loss1 + loss2
    else:
        if args.face_con_replace_diag:
            face_proto_match = torch.matmul(face_j_match.unsqueeze(1), face_j_add.permute(0, 2, 1))
            # replace diagonal elements with matched faces
            face_proto_match.diagonal().copy_(face_match.diagonal())
        else:
            face_proto_match = torch.matmul(face_j_match.unsqueeze(1), face_j_add.permute(0, 2, 1))
        loss1 = batch_softmax(face_proto_match, args.max_type)
        
        if args.face_con_replace_diag:
            proto_face_match = torch.matmul(face_j_add.unsqueeze(1), face_j_match.permute(0, 2, 1))
            # replace diagonal elements with matched faces
            proto_face_match.diagonal().copy_(face_match.diagonal())
        else:
            proto_face_match = torch.matmul(face_j_add.unsqueeze(1), face_j_match.permute(0, 2, 1))
        loss2 = batch_softmax(proto_face_match, args.max_type)

        loss3 = batch_agreement(face_proto_match, proto_face_match, args.agree_type, args.max_type)

        loss = loss1 + loss2 + args.alpha * loss3
    return loss


def prep_for_training(model, optimizer_type, train_size):
    model.to(DEVICE)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if optimizer_type == "adam":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_type == "lars":
        optimizer = LARS(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )
    elif optimizer_type == "sgd":
        optimizer = SGD(optimizer_grouped_parameters, args.lr, args.sgd_momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.lr_scheduler_tmax, eta_min=0, last_epoch=-1
    )

    return model, optimizer, scheduler

def train_incre_epoch(model, loss_type, frag_loss, global_loss, train_dataloader, one_data, real_bsz, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        face_emb, ner_features, ner_ids, word_emb = batch["face_emb"], batch["ner_features"], batch["ner_ids"], batch["word_emb"]

        sample_indices = random.sample(range(len(one_data)), real_bsz-args.batch_rest)
        one_one_samples = Subset(one_data, indices=sample_indices)

        face_emb_one_list = []
        ner_feat_one_list = []

        for sample in one_one_samples:
            face_emb_one_list.append(sample["face_emb"])
            ner_feat_one_list.append(sample["ner_features"])

        face_emb_one_pad = collate_tensors_by_size(face_emb_one_list, [1, face_emb.size(-2), face_emb.size(-1)])
        ner_feat_one_pad = collate_tensors_by_size(ner_feat_one_list, [1, ner_features.size(-2), ner_features.size(-1)])

        face_emb_final = torch.cat((face_emb, face_emb_one_pad), dim=0)
        ner_features_final = torch.cat((ner_features, ner_feat_one_pad), dim=0)

        if args.fine_tune is True:
            face_j, ner_j = model(face_emb.squeeze(1).to(DEVICE), ner_ids.to(DEVICE))
        elif args.use_onehot:
            face_j, ner_j = model(face_emb.squeeze(1).cuda(), word_emb.squeeze(1).cuda())
        else:
            face_j, ner_j = model(face_emb_final.squeeze(1).cuda(), ner_features_final.squeeze(1).cuda())

        face_j, ner_j = face_j.to(DEVICE), ner_j.to(DEVICE)

        if loss_type == "all":
            a = frag_loss(face_j, ner_j)
            b = global_loss(face_j, ner_j)
            loss = a + b
        else:
            loss = frag_loss(face_j, ner_j)

        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        else:
            loss = loss
        if args.use_onehot:
            loss.backward(retain_graph=True)
        else:
            loss.backward()


        tr_loss += loss.item()
        nb_tr_steps += 1

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 20 == 0 and not step == 0:
            print(f"step {step} / {len(train_dataloader)} | loss = {tr_loss / nb_tr_steps}")

    return tr_loss / nb_tr_steps


def collate_tensors_by_size(batch, max_face_size):
    dims_face_emb = batch[0].dim()
    size = (len(batch),) + tuple(max_face_size)

    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims_face_emb):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def train_all_incre(
        model,
        loss_type,
        frag_loss,
        global_loss,
        all_dataloader,
        optimizer,
        scheduler,
        one_data, real_bsz,
):
    train_losses = []

    for epoch_i in range(int(args.num_epoch)):
        train_loss = train_incre_epoch(model, loss_type, frag_loss, global_loss, all_dataloader, one_data, real_bsz, optimizer, scheduler)

        print(
            "epoch:{}, train_loss:{}".format(
                epoch_i, train_loss
            )
        )

        train_losses.append(train_loss)

    return train_losses


def train_incre_control_one_epoch(model, loss_type, frag_loss, global_loss, train_dataloader, unique_name_avg_face_dict, unique_name_dict, text_model, face_model, sample_type, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        face_emb, ner_features, ner_ids, word_emb, name_list = batch["face_emb"], batch["ner_features"], batch["ner_ids"], batch["word_emb"],  batch["names"]

        name_from_one_one = []
        for i in range(len(name_list)):
            for name in name_list[i]:
                if name[0] in unique_name_avg_face_dict.keys():
                    name_from_one_one.append(name[0])

        if len(name_from_one_one) > 0 and len(name_from_one_one) < args.batch_rest:
            if sample_type == "prototype":

                face_emb_one_list, ner_feat_one_list = prepare_prototypes(unique_name_avg_face_dict, list(set(name_from_one_one)), args.add_special_token, args.layer_start, args.layer_end, text_model, face_model, DEVICE)
            else:
                face_emb_one_list, ner_feat_one_list = prepare_samples(unique_name_dict, list(set(name_from_one_one)), args.add_special_token, args.layer_start, args.layer_end, text_model, face_model)
            face_emb_one_pad = collate_tensors_by_size(face_emb_one_list, [1, face_emb.size(-2), face_emb.size(-1)])
            ner_feat_one_pad = collate_tensors_by_size(ner_feat_one_list,
                                                       [1, ner_features.size(-2), ner_features.size(-1)])

            face_emb_final = torch.cat((face_emb, face_emb_one_pad), dim=0)
            ner_features_final = torch.cat((ner_features, ner_feat_one_pad), dim=0)
        elif len(name_from_one_one) > args.batch_rest:
            name_from_one_one = name_from_one_one[:args.batch_rest]
            if sample_type == "prototype":

                face_emb_one_list, ner_feat_one_list = prepare_prototypes(unique_name_avg_face_dict, list(set(name_from_one_one)), args.add_special_token, args.layer_start, args.layer_end, text_model, face_model, DEVICE)
            else:
                face_emb_one_list, ner_feat_one_list = prepare_samples(unique_name_dict, list(set(name_from_one_one)), args.add_special_token, args.layer_start, args.layer_end, text_model, face_model)
            face_emb_one_pad = collate_tensors_by_size(face_emb_one_list, [1, face_emb.size(-2), face_emb.size(-1)])
            ner_feat_one_pad = collate_tensors_by_size(ner_feat_one_list,
                                                       [1, ner_features.size(-2), ner_features.size(-1)])

            face_emb_final = torch.cat((face_emb, face_emb_one_pad), dim=0)
            ner_features_final = torch.cat((ner_features, ner_feat_one_pad), dim=0)
        else:
            face_emb_final = face_emb
            ner_features_final = ner_features

        if args.fine_tune is True:
            face_j, ner_j = model(face_emb.squeeze(1).to(DEVICE), ner_ids.to(DEVICE))
        elif args.use_onehot:
            face_j, ner_j = model(face_emb.squeeze(1).cuda(), word_emb.squeeze(1).cuda())
        else:
            face_j, ner_j = model(face_emb_final.squeeze(1).cuda(), ner_features_final.squeeze(1).cuda())

        face_j, ner_j = face_j.to(DEVICE), ner_j.to(DEVICE)

        if loss_type == "all":
            a = frag_loss(face_j, ner_j)
            b = global_loss(face_j, ner_j)
            loss = a + b
        else:
            loss = frag_loss(face_j, ner_j)

        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        else:
            loss = loss
        if args.use_onehot:
            loss.backward(retain_graph=True)
        else:
            loss.backward()


        tr_loss += loss.item()
        nb_tr_steps += 1

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 20 == 0 and not step == 0:
            print(f"step {step} / {len(train_dataloader)} | loss = {tr_loss / nb_tr_steps}")

    return tr_loss / nb_tr_steps



def train_all_control_one_incre(
        model, loss_type, frag_loss, global_loss, train_dataloader,
        unique_name_avg_face_dict, unique_name_dict, text_model, face_model,
        sample_type,
        optimizer, scheduler
):
    train_losses = []
    for epoch_i in range(int(args.num_epoch)):
        train_loss = train_incre_control_one_epoch(model, loss_type, frag_loss, global_loss, train_dataloader, unique_name_avg_face_dict, unique_name_dict, text_model, face_model, sample_type, optimizer, scheduler)
        print(
            "epoch:{}, train_loss:{}".format(
                epoch_i, train_loss
            )
        )

        train_losses.append(train_loss)

    return train_losses



def prepare_prototypes(unique_name_avg_face_dict, name_list, add_special_token, layer_start, layer_end, text_model, face_model, DEVICE):

    face_feat_list = []
    ner_feat_list = []

    for i, name in enumerate(name_list):
        avg_face = Image.open(unique_name_avg_face_dict[name]["avg_face_dir"]).convert("RGB")
        face_feat = face_model(transforms.ToTensor()(avg_face).to(DEVICE).unsqueeze_(0).to(DEVICE))
        face_feat_list.append(face_feat.unsqueeze(0))

        name_list_noname = [[name], ["NONAME"]]
        name_feat = gen_ner_emb_by_layer(tokenizer, text_model, name_list_noname, add_special_token, layer_start, layer_end)
        ner_feat_list.append(name_feat.unsqueeze(0))

    return face_feat_list, ner_feat_list


def prepare_samples(unique_name_dict, name_list, add_special_token, layer_start, layer_end,
                    text_model, face_model):

    face_feat_list = []
    ner_feat_list = []

    for i, name in enumerate(name_list):
        num_faces = len(unique_name_dict[name]["img_name"])
        if num_faces > 1:
            face_idx = random.randint(0, num_faces-1)
            face_features = prepare_sampled_face(face_model,
                                                 base_dir,
                                                 unique_name_dict[name],
                                                 face_idx,
                                                 160)
        else:
            face_features = prepare_sampled_face(face_model,
                                                 base_dir,
                                                 unique_name_dict[name],
                                                 0,
                                                 160)

        face_feat_list.append(face_features.unsqueeze(0))

        name_list_noname = [[name], ["NONAME"]]
        name_feat = gen_ner_emb_by_layer(tokenizer, text_model, name_list_noname, add_special_token, layer_start, layer_end)
        ner_feat_list.append(name_feat.unsqueeze(0))

    return face_feat_list, ner_feat_list


def prepare_sampled_face(face_model, base_dir, unique_name_dict, face_idx, out_face_size):
    img_name = unique_name_dict["img_dir"][face_idx]
    img_dir = os.path.join("/cw/working-rose/CelebTo/images_ct", img_name)

    face_bbox = unique_name_dict["bbox"][face_idx]

    img = Image.open(img_dir).convert("RGB")

    crop_face = crop_resize(img, face_bbox, out_face_size)  # size 3*160*160

    face_features = face_model(transforms.ToTensor()(crop_face).unsqueeze_(0).to("cuda"))

    return face_features


def gen_ner_emb_by_layer(tokenizer, text_model, ner_list, add_special_token, layer_start, layer_end):
    len_ner = len(ner_list)
    ner_features = torch.empty(len_ner, 768)
    for i in range(len_ner):
        ner = ner_list[i][0]

        encoded_ids = tokenizer.encode_plus(text=ner,
                                            add_special_tokens=add_special_token,
                                            return_tensors="pt")["input_ids"]

        ner_id = encoded_ids.to("cuda")
        if add_special_token:
            ner_emb = torch.mean(sum(text_model(ner_id)["hidden_states"][layer_start:layer_end])[:, 1:-1, :], dim=1)
        else:
            ner_emb = torch.mean(sum(text_model(ner_id)["hidden_states"][layer_start:layer_end]), dim=1)
        ner_features[i] = ner_emb.squeeze()

    return ner_features


class ZeroPadCollator:
    @staticmethod
    def collate_tensors(batch) -> torch.Tensor:
        dims_face_emb = batch[0].dim()
        max_face_size = [max([b.size(i) for b in batch]) for i in range(dims_face_emb)]
        # print("max_face_size:{}".format(max_face_size))
        size = (len(batch),) + tuple(max_face_size)

        canvas = batch[0].new_zeros(size=size)
        for i, b in enumerate(batch):
            sub_tensor = canvas[i]
            for d in range(dims_face_emb):
                sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
            sub_tensor.add_(b)
        return canvas

    @staticmethod
    def split_batch(batch, key):
        out_list = []
        for i in range(len(batch)):
            out_list.append(batch[i][key])
        return out_list

    def collate_fn(self, batch):
        num_faces = self.split_batch(batch, "num_faces")
        face_tensors = self.split_batch(batch, "face_tensor")
        face_features = self.split_batch(batch, "face_emb")
        # caption_raw = self.split_batch(batch, "caption_raw")
        # caption_ids = self.split_batch(batch, "caption_ids")
        ner_ids = self.split_batch(batch, "ner_ids")
        caption_emb = self.split_batch(batch, "caption_emb")
        img_rgb = self.split_batch(batch, "img_rgb")
        names = self.split_batch(batch, "names")
        ner_list = self.split_batch(batch, "ner_list")
        ner_features = self.split_batch(batch, "ner_features")
        ner_context_features = self.split_batch(batch, "ner_context_features")
        gt_link = self.split_batch(batch, "gt_link")
        word_emb = self.split_batch(batch, "word_emb")

        pad_ner_ids = self.collate_tensors(ner_ids)
        pad_face_tensors = self.collate_tensors(face_tensors)
        pad_face_features = self.collate_tensors(face_features)
        # pad_caption_emb = self.collate_tensors(caption_emb)
        pad_ner_features = self.collate_tensors(ner_features)
        # pad_ner_context_features = self.collate_tensors(ner_context_features)
        # pad_word_emb = self.collate_tensors(word_emb)

        return {
            "num_faces": num_faces,
            "face_tensor": pad_face_tensors,
            # "face_tensor": [self.collate_tensors([b["face_tensor"] for b in batch])],
            "face_emb": pad_face_features,
            "caption_raw": {},
            "caption_ids": {},
            "ner_ids": pad_ner_ids,
            "caption_emb": {},
            "img_rgb": img_rgb,
            "names": names,
            "ner_list": ner_list,
            "ner_features": pad_ner_features,
            "ner_context_features": {},
            # "name_ids": name_ids,
            "word_emb": {},
        }


def batch_softmax_nomax(phrase_region_match):
    # phrase_region_match [B, B, span_len1, span_len2]: span_len1: names, span_len2: faces
    batch_size, _, num_spans1, num_spans2 = phrase_region_match.size()

    # [B, B, span_len1]
    phrase_region_sum = phrase_region_match.sum(-1)

    phrase_region_sum_norm = phrase_region_sum.div(
        torch.tensor(num_spans2, device=phrase_region_sum.device).expand(batch_size).unsqueeze(1).expand(phrase_region_sum.size())
    )

    # Logits [B, B]
    phrase_region_scores = phrase_region_sum_norm.sum(-1)
    # Normalize scores
    logits = phrase_region_scores.div(
        torch.tensor(num_spans1, device=phrase_region_scores.device).expand(batch_size).unsqueeze(1).expand(phrase_region_scores.size())
    )
    targets = torch.arange(
        batch_size, device=phrase_region_scores.device
    )

    return F.cross_entropy(logits, targets)

class BatchSoftmaxNomax(nn.Module):
    def __init__(self):
        super(BatchSoftmaxNomax, self).__init__()

    def forward(self, face_j, ner_j):
        face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
        return batch_softmax_nomax(face_ner_match)


def get_sim_scores(model, all_faces, ner_pos_i, add_extra_proj, proj_type, DEVICE):
    num_face_i = all_faces.size()[1]
    face_list_all = []
    for j in range(num_face_i):

        if add_extra_proj:

            face_z_i = model.model_one_stage.projector(all_faces.squeeze(0)[j])
            if face_z_i.dim() < 1:
                face_z_i = face_z_i.unsqueeze(0)

            ner_i = ner_pos_i

            ner_z_j = model.model_one_stage.ner_proj(ner_i.squeeze(0).to(DEVICE))

            if proj_type == "one":
                ner_z_all = model.model_one_stage.projector(ner_z_j)
            else:
                ner_z_all = model.model_one_stage.ner_projector(ner_z_j)
        else:
            face_z_i = model.projector(all_faces.squeeze(0)[j])
            if face_z_i.dim() < 1:
                face_z_i = face_z_i.unsqueeze(0)

            ner_i = ner_pos_i

            if proj_type == "one":
                ner_z_all = model.projector(model.ner_proj(ner_i.squeeze(0).to(DEVICE)))
            else:
                ner_z_all = model.ner_projector(model.ner_proj(ner_i.squeeze(0).to(DEVICE)))

        sim_all = torch.matmul(face_z_i, torch.transpose(ner_z_all, 0, 1))
        face_list_all.append(sim_all.tolist())
    return face_list_all

def find_aligned_names(sim_lists, name_list):
    aligned_names = []
    for sim_list in sim_lists:
        if max(sim_list) > 0:
            aligned_name = name_list[sim_list.index(max(sim_list))]
            aligned_names.append(aligned_name)
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


def batch_match_incre(model, add_sample,
                      text_model, face_model,
                      add_extra_proj, proj_type, batch,
                      dict_one_unique, unique_name_avg_face_dict):
    face_emb, ner_features, name_list, num_faces = batch["face_emb"], batch["ner_features"], batch["names"], batch["num_faces"]

    face_unmatch_list = []
    name_unmatch_list = []

    face_match_list = []
    name_match_list = []

    matched_names = []
    num_matched_names = []

    for i in range(len(name_list)):
        name_from_one_one = []
        for name in name_list[i]:
            if name[0] in dict_one_unique.keys():
                name_from_one_one.append(name[0])
        # print("{}th name list:{}".format(i, name_from_one_one))

        # de-padding face_emb
        face_emb_orginal = face_emb[i][:, :num_faces[i], :].to("cuda")
        # de-padding ner_features
        ner_feat_original = ner_features[i][:, :len(name_list[i]), :].to("cuda")

        sim_scores = get_sim_scores(model, face_emb_orginal, ner_feat_original, add_extra_proj, proj_type, "cuda")

        # print(sim_scores)
        aligned_names = find_aligned_names(sim_scores, name_list[i])
        # print(aligned_names)
        # aligned_names_unique = list(set(aligned_names))
        #
        aligned_names_unique = unique([name[0] for name in aligned_names])
        # print(aligned_names_unique)

        deleted_face_idx = []
        deleted_name_idx = []
        matched_name_one_sample = []

        if len(name_from_one_one) == 0:
            face_unmatch_list.append(face_emb_orginal.to("cuda"))
            name_unmatch_list.append(ner_feat_original.to("cuda"))
        else:
            for idx, name in enumerate(name_from_one_one):
                if name in aligned_names_unique:

                    name_aligned_count = Counter([name[0] for name in aligned_names])
                    name_index = name_list[i].index([name])

                    aligned_idx = duplicates_idx(aligned_names, [name])

                    aligned_sim_scores = [sim_scores[i] for i in aligned_idx]
                    name_associate_scores = [score[name_index] for score in aligned_sim_scores]

                    if max(name_associate_scores) < 0:
                        continue

                    elif name_aligned_count[name] > 1:
                        # find index of the name in name list
                        # print("name list:{}".format(name_list[i]))
                        # find max sim score at the index
                        max_index = name_associate_scores.index(max(name_associate_scores))

                        deleted_face_idx.append(aligned_idx[max_index])
                        deleted_name_idx.append(name_index)

                        matched_name_one_sample.append(name)

                    else:
                        deleted_face_idx.append(aligned_names.index([name]))
                        deleted_name_idx.append(name_list[i].index([name]))
                        matched_name_one_sample.append(name)
                else:
                    continue

            # print(deleted_face_idx)
            # print(deleted_name_idx)
            num_matched_names.append(len(deleted_name_idx))

            if len(deleted_face_idx) > 0 and len(deleted_name_idx) > 0:
                # extract matched face features
                face_match_feat = torch.index_select(face_emb_orginal, 1, torch.LongTensor(deleted_face_idx).to("cuda"))
                # extract matched name features
                ner_match_feat = torch.index_select(ner_feat_original,1, torch.LongTensor(deleted_name_idx).to("cuda"))

                unmatched_idx_face = list(set(range(num_faces[i])) - set(deleted_face_idx))
                unmatched_idx_name = list( set(range(len(name_list[i]))) - set(deleted_name_idx))

                face_unmatch_feat = torch.index_select(face_emb_orginal, 1, torch.LongTensor(unmatched_idx_face).to("cuda"))
                ner_unmatch_feat = torch.index_select(ner_feat_original, 1, torch.LongTensor(unmatched_idx_name).to("cuda"))

                face_match_list.append(face_match_feat)
                name_match_list.append(ner_match_feat)

                face_unmatch_list.append(face_unmatch_feat)
                name_unmatch_list.append(ner_unmatch_feat)
            else:
                face_unmatch_list.append(face_emb_orginal.to("cuda"))
                name_unmatch_list.append(ner_feat_original.to("cuda"))

            if len(matched_name_one_sample) > 0:
                matched_names.append(matched_name_one_sample)

    # add samples to match/unmatch list if needed
    if args.add_to == "match":
        # in match list, we add NONAME i.e. "False", ok for max, should do the other way for nomax
        if add_sample == "prototype":
            face_match_list, name_match_list = add_prototypes(unique_name_avg_face_dict, False, matched_names, name_match_list, face_match_list, True, -4, None,
            text_model, face_model, "cuda")
        elif add_sample == "random":
            face_match_list, name_match_list = add_random_sample(unique_name_dict, False, matched_names, name_match_list, face_match_list, True, -4,
            None, text_model, face_model)
        else:
            face_match_list = face_match_list
            name_match_list = name_match_list
    else:
        # in unmatch list, we do not add noname  --> disambiguition
        if add_sample == "prototype":
            face_unmatch_list, name_unmatch_list = add_prototypes(unique_name_avg_face_dict, True, matched_names, name_unmatch_list, face_unmatch_list, True, -4, None, text_model, face_model, "cuda")
        elif add_sample == "random":
            face_unmatch_list, name_unmatch_list = add_random_sample(unique_name_dict, True, matched_names, name_unmatch_list, face_unmatch_list, True, -4,
            None, text_model, face_model)
        else:
            face_unmatch_list = face_unmatch_list
            name_unmatch_list = name_unmatch_list

    if len(face_match_list) > 0:
        max_faces_match = max([face_emb.size()[1]  for face_emb in face_match_list])
        max_names_match = max([ner_feat.size()[1]  for ner_feat in name_match_list])

        max_faces_unmatch = max([face_emb.size()[1] for face_emb in face_unmatch_list])
        max_names_unmatch = max([ner_feat.size()[1] for ner_feat in name_unmatch_list])

        if max_faces_unmatch == 0:
            # in case of all one-one pairs
            face_match_emb = collate_tensors_by_size(face_match_list, [1, max_faces_match, 512])
            name_match_emb = collate_tensors_by_size(name_match_list, [1, max_names_match, 768])

            face_unmatch_emb = None
            name_unmatch_emb = None
        else:
            face_match_emb = collate_tensors_by_size(face_match_list, [1, max_faces_match, 512])
            name_match_emb = collate_tensors_by_size(name_match_list, [1, max_names_match, 768])

            face_unmatch_emb = collate_tensors_by_size(face_unmatch_list, [1, max_faces_unmatch, 512])
            name_unmatch_emb = collate_tensors_by_size(name_unmatch_list, [1, max_names_unmatch, 768])

    else:
        face_match_emb = None
        name_match_emb = None
        max_faces_unmatch = max([face_emb.size()[1] for face_emb in face_unmatch_list])
        max_names_unmatch = max([ner_feat.size()[1] for ner_feat in name_unmatch_list])

        face_unmatch_emb = collate_tensors_by_size(face_unmatch_list, [1, max_faces_unmatch, 512])
        name_unmatch_emb = collate_tensors_by_size(name_unmatch_list, [1, max_names_unmatch, 768])

    return face_match_emb, name_match_emb, face_unmatch_emb, name_unmatch_emb, num_matched_names


def add_prototypes(unique_name_avg_face_dict, nomax, matched_names, name_match_list, face_match_list, add_special_token, layer_start, layer_end, text_model, face_model, DEVICE):
    if args.add_to == "match":
        for i, matched_list in enumerate(matched_names):
            face_feat_add = torch.empty((len(matched_list), 512))
            for j, name in enumerate(matched_list):
                avg_face = Image.open(unique_name_avg_face_dict[name]["avg_face_dir"]).convert("RGB")
                face_feat = face_model(transforms.ToTensor()(avg_face).to(DEVICE).unsqueeze_(0).to(DEVICE))
                face_feat_add[j] = face_feat

            if nomax:
                face_match_list[i] = torch.cat((face_match_list[i], face_feat_add.unsqueeze(0).to("cuda")), dim=1)
            else:
                noname_feat = gen_ner_emb_by_layer(tokenizer, text_model, [["NONAME"]], add_special_token, layer_start, layer_end)

                face_match_list[i] = torch.cat((face_match_list[i], face_feat_add.unsqueeze(0).to("cuda")), dim=1)
                name_match_list[i] = torch.cat((name_match_list[i], noname_feat.unsqueeze(0).to("cuda")), dim=1)
    else:
        # directly append to unmatch list
        matched_names_unique = unique(matched_names)
        for i, matched_list in enumerate(matched_names_unique):
            face_feat_add = torch.empty((len(matched_list), 512))
            for j, name in enumerate(matched_list):
                avg_face = Image.open(unique_name_avg_face_dict[name]["avg_face_dir"]).convert("RGB")
                face_feat = face_model(transforms.ToTensor()(avg_face).to(DEVICE).unsqueeze_(0).to(DEVICE))
                face_feat_add[j] = face_feat
            
            face_match_list.append(face_feat_add.unsqueeze(0).to("cuda"))
            # make format of matched list suitable for gen_ner_emb_by_layer 
            name_list = [[name] for name in matched_list]
            name_feat = gen_ner_emb_by_layer(tokenizer, text_model, name_list, add_special_token, layer_start, layer_end)
            name_match_list.append(name_feat.unsqueeze(0).to("cuda"))

    return face_match_list, name_match_list


def add_random_sample(unique_name_dict, nomax, matched_names, name_match_list, face_match_list, add_special_token, layer_start, layer_end, text_model, face_model):
    if args.add_to == "match":
        for i, matched_list in enumerate(matched_names):
            face_feat_add = torch.empty((len(matched_list), 512))
            for j, name in enumerate(matched_list):
                num_faces = len(unique_name_dict[name]["img_name"])
                if num_faces > 1:
                    face_idx = random.randint(0, num_faces-1)
                    face_feat = prepare_sampled_face(face_model, base_dir,
                    unique_name_dict[name], face_idx, 160)
                    face_feat_add[j] = face_feat
                else:
                    face_feat = prepare_sampled_face(face_model, base_dir,
                    unique_name_dict[name], 0, 160)
                    face_feat_add[j] = face_feat

            if nomax:
                face_match_list[i] = torch.cat((face_match_list[i], face_feat_add.unsqueeze(0).to("cuda")), dim=1)
            else:
                noname_feat = gen_ner_emb_by_layer(tokenizer, text_model, [["NONAME"]], add_special_token, layer_start, layer_end)
                face_match_list[i] = torch.cat((face_match_list[i], face_feat_add.unsqueeze(0).to("cuda")), dim=1)
                name_match_list[i] = torch.cat((name_match_list[i], noname_feat.unsqueeze(0).to("cuda")), dim=1)
    else:
        matched_names_unique = unique(matched_names)
        for i, matched_list in enumerate(matched_names_unique):
            face_feat_add = torch.empty((len(matched_list), 512))
            for j, name in enumerate(matched_list):
                num_faces = len(unique_name_dict[name]["img_name"])
                if num_faces > 1:
                    face_idx = random.randint(0, num_faces-1)
                    face_feat = prepare_sampled_face(face_model, base_dir,
                    unique_name_dict[name], face_idx, 160)
                    face_feat_add[j] = face_feat
                else:
                    face_feat = prepare_sampled_face(face_model, base_dir,
                    unique_name_dict[name], 0, 160)
                    face_feat_add[j] = face_feat
            
            face_match_list.append(face_feat_add.unsqueeze(0).to("cuda"))
            # make format of matched list suitable for gen_ner_emb_by_layer 
            name_list = [[name] for name in matched_list]
            name_feat = gen_ner_emb_by_layer(tokenizer, text_model, name_list, add_special_token, layer_start, layer_end)
            name_match_list.append(name_feat.unsqueeze(0).to("cuda"))

    return face_match_list, name_match_list


def check_names_d_one(name_list):
    if ["NONAME"] in name_list and len(name_list) == 2:
        return True
    else:
        return False


def batch_match_incre_make_new(model, add_sample, text_model, face_model, add_extra_proj, proj_type, batch, dict_one_unique, unique_name_avg_face_dict, null_face):
    face_emb, ner_features, name_list, num_faces = batch["face_emb"], batch["ner_features"], batch["names"], batch["num_faces"]

    face_unmatch_list = []
    name_unmatch_list = []

    face_match_list = []
    name_match_list = []

    matched_names = []
    num_matched_names = []

    for i in range(len(name_list)):
        name_from_one_one = []
        for name in name_list[i]:
            if name in dict_one_unique.keys():
                name_from_one_one.append(name)
        # print("{}th name list:{}".format(i, name_from_one_one))

        # de-padding face_emb
        face_emb_original = face_emb[i][:, :num_faces[i], :].to("cuda")
        # de-padding ner_features
        ner_feat_original = ner_features[i][:, :len(name_list[i]), :].to("cuda")

        sim_scores = get_sim_scores(model, face_emb_original, ner_feat_original, add_extra_proj, proj_type, "cuda")

        # print(sim_scores)
        aligned_names = find_aligned_names(sim_scores, name_list[i])
        # print(aligned_names)
        # aligned_names_unique = list(set(aligned_names))
        #
        aligned_names_unique = unique([name for name in aligned_names])
        # print(aligned_names_unique)

        deleted_face_idx = []
        deleted_name_idx = []
        matched_name_one_sample = []

        if len(name_from_one_one) == 0:
            face_unmatch_list.append(face_emb_original.to("cuda"))
            name_unmatch_list.append(ner_feat_original.to("cuda"))
        elif args.add_d_one and num_faces[i] == 1 and check_names_d_one(name_list[i]):
            # print(name_list[i])
            # print(face_emb_original.size())
            face_unmatch_list.append(face_emb_original.to("cuda"))
            name_unmatch_list.append(ner_feat_original.to("cuda"))
        else:
            for idx, name in enumerate(name_from_one_one):
                if name in aligned_names_unique:

                    name_aligned_count = Counter([name for name in aligned_names])
                    name_index = name_list[i].index(name)

                    aligned_idx = duplicates_idx(aligned_names, name)

                    aligned_sim_scores = [sim_scores[i] for i in aligned_idx]
                    name_associate_scores = [score[name_index] for score in aligned_sim_scores]

                    if max(name_associate_scores) < 0:
                        continue

                    elif name_aligned_count[name] > 1:
                        # find index of the name in name list
                        # print("name list:{}".format(name_list[i]))
                        # find max sim score at the index
                        max_index = name_associate_scores.index(max(name_associate_scores))

                        deleted_face_idx.append(aligned_idx[max_index])
                        deleted_name_idx.append(name_index)

                        matched_name_one_sample.append(name)

                    else:
                        deleted_face_idx.append(aligned_names.index(name))
                        deleted_name_idx.append(name_list[i].index(name))
                        matched_name_one_sample.append(name)
                else:
                    continue

            # print(deleted_face_idx)
            # print(deleted_name_idx)
            num_matched_names.append(len(deleted_name_idx))

            if len(deleted_face_idx) > 0 and len(deleted_name_idx) > 0:
                # extract matched face features
                face_match_feat = torch.index_select(face_emb_original, 1, torch.LongTensor(deleted_face_idx).to("cuda"))
                # extract matched name features
                ner_match_feat = torch.index_select(ner_feat_original,1, torch.LongTensor(deleted_name_idx).to("cuda"))

                unmatched_idx_face = list(set(range(num_faces[i])) - set(deleted_face_idx))
                unmatched_idx_name = list( set(range(len(name_list[i]))) - set(deleted_name_idx))

                face_unmatch_feat = torch.index_select(face_emb_original, 1, torch.LongTensor(unmatched_idx_face).to("cuda"))
                ner_unmatch_feat = torch.index_select(ner_feat_original, 1, torch.LongTensor(unmatched_idx_name).to("cuda"))

                if args.add_nullface:
                    face_match_feat = torch.cat((face_match_feat.squeeze(0), null_face))
                    face_match_feat = face_match_feat.unsqueeze(0)
                else:
                    face_match_feat = face_match_feat

                face_match_list.append(face_match_feat)
                name_match_list.append(ner_match_feat)

                face_unmatch_list.append(face_unmatch_feat)
                name_unmatch_list.append(ner_unmatch_feat)
            else:
                face_unmatch_list.append(face_emb_original.to("cuda"))
                name_unmatch_list.append(ner_feat_original.to("cuda"))

            if len(matched_name_one_sample) > 0:
                matched_names.append(matched_name_one_sample)

    if add_sample == "prototype":
        face_add_list, name_match_list = make_prototypes(unique_name_avg_face_dict, args.noname_to_match, matched_names, name_match_list, True, -4, None, text_model, face_model, null_face, "cuda")
    elif add_sample == "near":
        face_add_list, name_match_list = make_nearest_face(model, dict_one_unique, args.noname_to_match, matched_names, name_match_list, True, -4, None, text_model, face_model, null_face, 160, 128, "cuda")
    else:
        # add_sample == random, mid-sim, mid-pdist
        face_add_list, name_match_list = make_random_faces(dict_one_unique, args.noname_to_match, matched_names, name_match_list, True, -4, None, text_model, face_model, null_face, "cuda")
    

    if len(face_match_list) > 0:
        max_faces_match = max([face_emb.size()[1]  for face_emb in face_match_list])
        max_names_match = max([ner_feat.size()[1]  for ner_feat in name_match_list])

        max_faces_unmatch = max([face_emb.size()[1] for face_emb in face_unmatch_list])
        max_names_unmatch = max([ner_feat.size()[1] for ner_feat in name_unmatch_list])

        if max_faces_unmatch == 0:
            # in case of all one-one pairs
            face_match_emb = collate_tensors_by_size(face_match_list, [1, max_faces_match, 512])
            name_match_emb = collate_tensors_by_size(name_match_list, [1, max_names_match, 768])
            face_add_emb = collate_tensors_by_size(face_add_list, [1, max_faces_match, 512])

            face_unmatch_emb = None
            name_unmatch_emb = None
        else:
            face_match_emb = collate_tensors_by_size(face_match_list, [1, max_faces_match, 512])
            name_match_emb = collate_tensors_by_size(name_match_list, [1, max_names_match, 768])
            face_add_emb = collate_tensors_by_size(face_add_list, [1, max_faces_match, 512])

            face_unmatch_emb = collate_tensors_by_size(face_unmatch_list, [1, max_faces_unmatch, 512])
            name_unmatch_emb = collate_tensors_by_size(name_unmatch_list, [1, max_names_unmatch, 768])

    else:
        face_match_emb = None
        name_match_emb = None
        face_add_emb = None
        max_faces_unmatch = max([face_emb.size()[1] for face_emb in face_unmatch_list])
        max_names_unmatch = max([ner_feat.size()[1] for ner_feat in name_unmatch_list])

        face_unmatch_emb = collate_tensors_by_size(face_unmatch_list, [1, max_faces_unmatch, 512])
        name_unmatch_emb = collate_tensors_by_size(name_unmatch_list, [1, max_names_unmatch, 768])

    return face_match_emb, face_add_emb, name_match_emb, face_unmatch_emb, name_unmatch_emb, num_matched_names


def make_prototypes(unique_name_avg_face_dict, nomax, matched_names, name_match_list, add_special_token, layer_start, layer_end, text_model, face_model, null_face, DEVICE):
    # make prototype faces list for each matched names
    face_proto_list = []
    for i, matched_list in enumerate(matched_names):
        if args.add_nullface and len(matched_list)>0:
            face_feat_add = torch.empty((len(matched_list)+1, 512))
        else:
            face_feat_add = torch.empty((len(matched_list), 512))
        for j, name in enumerate(matched_list):
            avg_face = Image.open(unique_name_avg_face_dict[name]["avg_face_dir"]).convert("RGB")
            face_feat = face_model(transforms.ToTensor()(avg_face).to(DEVICE).unsqueeze_(0).to(DEVICE))
            face_feat_add[j] = face_feat

        if args.add_nullface and len(matched_list)>0:
            face_feat_add[-1] = null_face
        else:
            face_feat_add = face_feat_add

        if nomax:
            face_proto_list.append(face_feat_add.unsqueeze(0))
        else:
            noname_feat = gen_ner_emb_by_layer(tokenizer, text_model, [["NONAME"]], add_special_token, layer_start, layer_end)
            face_proto_list.append(face_feat_add.unsqueeze(0))
            name_match_list[i] = torch.cat((name_match_list[i], noname_feat.unsqueeze(0).to("cuda")), dim=1)

    return face_proto_list, name_match_list


def make_random_faces(unique_name_dict, nomax, matched_names, name_match_list,  add_special_token, layer_start, layer_end, text_model, face_model, null_face, DEVICE):
    # make random faces list for each matched names
    face_random_list = []
    for i, matched_list in enumerate(matched_names):
        if args.add_nullface and len(matched_list)>0:
            face_feat_add = torch.empty((len(matched_list)+1, 512))
        else:
            face_feat_add = torch.empty((len(matched_list), 512))
        for j, name in enumerate(matched_list):
            num_faces = len(unique_name_dict[name]["img_dir"])
            if num_faces > 1:
                face_idx = random.randint(0, num_faces-1)
                face_feat = prepare_sampled_face(face_model, base_dir,
                unique_name_dict[name], face_idx, 160)
                face_feat_add[j] = face_feat
            else:
                face_feat = prepare_sampled_face(face_model, base_dir,
                unique_name_dict[name], 0, 160)
                face_feat_add[j] = face_feat
        
        if args.add_nullface and len(matched_list)>0:
            face_feat_add[-1] = null_face
        else:
            face_feat_add = face_feat_add

        if nomax:
            face_random_list.append(face_feat_add.unsqueeze(0))
        else:
            noname_feat = gen_ner_emb_by_layer(tokenizer, text_model, [["NONAME"]], add_special_token, layer_start, layer_end)
            face_random_list.append(face_feat_add.unsqueeze(0)) 
            name_match_list[i] = torch.cat((name_match_list[i], noname_feat.unsqueeze(0).to("cuda")), dim=1)

    return face_random_list, name_match_list


def get_all_face(face_dict, name, DEVICE, face_model, stage1_model):
    """get face tensors based on bounding box"""
    # face_tensors = torch.load(face_dict[name]["face_tensor"]).to(DEVICE)
    facenet_tensors = torch.load(face_dict[name]["facenet_tensor"]).to(DEVICE)
    
    # face_features = stage1_model.projector(face_model(face_tensors))
    face_features = stage1_model.projector(facenet_tensors)

    return face_features, facenet_tensors


def make_nearest_face(stage1_model, unique_name_dict, nomax, matched_names, name_match_list, add_special_token, layer_start, layer_end, text_model, face_model, null_face, out_face_size, face_feat_size, DEVICE):
    # make prototype faces list for each matched names
    face_proto_list = []
    if not nomax:
        noname_feat = gen_ner_emb_by_layer(tokenizer, text_model, [["NONAME"]], add_special_token, layer_start, layer_end)
    # with torch.no_grad():
    for i, matched_list in enumerate(matched_names):
        if args.add_nullface:
            face_feat_add = torch.empty((len(matched_list)+1, 512))
        else:
            face_feat_add = torch.empty((len(matched_list), 512))
        for j, name in enumerate(matched_list):
            face_features, facenet_tensors = get_all_face(unique_name_dict, name, DEVICE, face_model, stage1_model)
            name_feat_z_i = stage1_model.ner_projector(stage1_model.ner_proj(name_match_list[i].squeeze(0)[j]))
            face_idx = torch.argmax(torch.matmul(name_feat_z_i.unsqueeze(0), face_features.to(DEVICE).t()))
            del face_features

            face_feat_add[j] = facenet_tensors[face_idx].unsqueeze(0)
            
        if args.add_nullface:
            face_feat_add[-1] = null_face
        else:
            face_feat_add = face_feat_add

        if nomax:
            face_proto_list.append(face_feat_add.unsqueeze(0))
        else:
            face_proto_list.append(face_feat_add.unsqueeze(0))
            name_match_list[i] = torch.cat((name_match_list[i], noname_feat.unsqueeze(0).to("cuda")), dim=1)
    
    return face_proto_list, name_match_list




def train_incre_split_sum_epoch(model, text_model, face_model, sum_loss, frag_loss, train_dataloader, unique_name_dict, unique_name_avg_face_dict, null_face, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    matched_names_size_list = []
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        if args.make_new:
            face_match_emb, face_add_emb, name_match_emb, face_unmatch_emb, name_unmatch_emb, num_matched_names = batch_match_incre_make_new(model, args.add_sample, text_model.cuda(), face_model.cuda(), args.add_extra_proj, args.proj_type, batch, unique_name_dict, unique_name_avg_face_dict, null_face)

            if face_add_emb is not None:
                face_j_match, ner_j_match = model(face_match_emb.squeeze(1).cuda(), name_match_emb.squeeze(1).cuda())

                face_j_unmatch, ner_j_unmatch = model(face_unmatch_emb.squeeze(1).cuda(), name_unmatch_emb.squeeze(1).cuda())

                face_j_add, ner_j_match = model(face_add_emb.squeeze(1).cuda(), name_match_emb.squeeze(1).cuda())

                loss1 = frag_loss(face_j_unmatch, ner_j_unmatch)
                
                if args.only_face_con and args.match_proto_agree:
                    loss3 = face_contras_loss(face_j_match, face_j_add, args.face_con_direction)
                    loss4 = frag_loss(face_j_add, ner_j_match)
                    loss = loss1 + loss3 + loss4
                elif args.only_face_con and not args.match_proto_agree:
                    loss3 = face_contras_loss(face_j_match, face_j_add, args.face_con_direction)
                    loss = loss1 + loss3
                elif args.face_contras and args.match_proto_agree:
                    loss2 = combined_frag_loss(face_j_match, face_j_add, ner_j_match, args.combined_direction)
                    loss3 = face_contras_loss(face_j_match, face_j_add, args.face_con_direction)
                    loss4 = frag_loss(face_j_add, ner_j_match)
                    loss = loss1 + loss2 + loss3 + loss4
                elif args.face_contras:
                    loss2 = combined_frag_loss(face_j_match, face_j_add, ner_j_match, args.combined_direction)
                    loss3 = face_contras_loss(face_j_match, face_j_add, args.face_con_direction)
                    loss = loss1 + loss2 + loss3
                elif not args.face_contras and args.match_proto_agree:
                    loss2 = combined_frag_loss(face_j_match, face_j_add, ner_j_match, args.combined_direction)
                    loss4 = frag_loss(face_j_add, ner_j_match)
                    loss = loss1 + loss2 + loss4
                else:
                    loss2 = combined_frag_loss(face_j_match, face_j_add, ner_j_match, args.combined_direction)
                    loss = loss1 + loss2

            elif face_unmatch_emb is None:
                face_j_match, ner_j_match = model(face_match_emb.squeeze(1).cuda(), name_match_emb.squeeze(1).cuda())
                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0
                if args.nomax:
                    loss = sum_loss(face_j_match, ner_j_match)
                else:
                    loss = frag_loss(face_j_match, ner_j_match)
            else:
                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0
                face_j_unmatch, ner_j_unmatch = model(face_unmatch_emb.squeeze(1).cuda(), name_unmatch_emb.squeeze(1).cuda())
                loss = frag_loss(face_j_unmatch, ner_j_unmatch)
        else:
            face_match_emb, name_match_emb, face_unmatch_emb, name_unmatch_emb, num_matched_names = batch_match_incre(model, args.add_sample, text_model.cuda(), face_model.cuda(), args.add_extra_proj, args.proj_type, batch, unique_name_dict, unique_name_avg_face_dict)

            if face_match_emb is not None:
                face_j_match, ner_j_match = model(face_match_emb.squeeze(1).cuda(), name_match_emb.squeeze(1).cuda())

                face_j_unmatch, ner_j_unmatch = model(face_unmatch_emb.squeeze(1).cuda(), name_unmatch_emb.squeeze(1).cuda())

                face_j_match, ner_j_match = face_j_match.to(DEVICE), ner_j_match.to(DEVICE)
                face_j_unmatch, ner_j_unmatch = face_j_unmatch.to(DEVICE), ner_j_unmatch.to(DEVICE)

                loss1 = frag_loss(face_j_unmatch, ner_j_unmatch)
                if args.nomax:
                    loss2 = sum_loss(face_j_match, ner_j_match)
                else:
                    loss2 = frag_loss(face_j_match, ner_j_match)

                loss = loss1 + loss2
            elif face_unmatch_emb is None:
                face_j_match, ner_j_match = model(face_match_emb.squeeze(1).cuda(), name_match_emb.squeeze(1).cuda())
                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0
                if args.nomax:
                    loss = sum_loss(face_j_match, ner_j_match)
                else:
                    loss = frag_loss(face_j_match, ner_j_match)
            else:
                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0
                face_j_unmatch, ner_j_unmatch = model(face_unmatch_emb.squeeze(1).cuda(), name_unmatch_emb.squeeze(1).cuda())
                loss = frag_loss(face_j_unmatch, ner_j_unmatch)

        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        else:
            loss = loss
        if args.use_onehot:
            loss.backward(retain_graph=True)
        else:
            loss.backward()

        tr_loss += loss.item()
        nb_tr_steps += 1

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        matched_names_size_list.append(num_matched_names)
        # print(f"{loss},{loss1},{loss2}")

        if step % 500 == 0 and not step == 0:
            if args.only_face_con and args.match_proto_agree:
                logger.info(f"step {step} / {len(train_dataloader)} | loss = {tr_loss / nb_tr_steps} | current frag loss = {loss1} | face contras loss = {loss3} | match frag loss = {loss4}")
            elif args.only_face_con and not args.match_proto_agree:
                logger.info(f"step {step} / {len(train_dataloader)} | loss = {tr_loss / nb_tr_steps} | current frag loss = {loss1} | face contras loss = {loss3}")
            elif not args.face_contras and not args.match_proto_agree:
                logger.info(f"step {step} / {len(train_dataloader)} | loss = {tr_loss / nb_tr_steps} | current frag loss = {loss1} | matched frag loss = {loss2}")
            elif not args.face_contras and args.match_proto_agree:
                logger.info(f"step {step} / {len(train_dataloader)} | loss = {tr_loss / nb_tr_steps} | current frag loss = {loss1} | matched frag loss = {loss2} | match frag loss = {loss4}")
            elif args.face_contras and args.match_proto_agree:
                logger.info(f"step {step} / {len(train_dataloader)} | loss = {tr_loss / nb_tr_steps} | current frag loss = {loss1} | matched frag loss = {loss2}| face contras loss = {loss3} | match frag loss = {loss4}")
            else:
                logger.info(f"step {step} / {len(train_dataloader)} | loss = {tr_loss / nb_tr_steps} | current frag loss = {loss1} | matched frag loss = {loss2} | face contras loss = {loss3}")

    text_model.to("cpu")
    face_model.to("cpu")
    return tr_loss / nb_tr_steps, matched_names_size_list


def train_all_incre_split_sum(model, text_model, face_model, sum_loss, frag_loss, train_dataloader,unique_name_dict, unique_name_avg_face_dict, null_face, optimizer, scheduler, out_dir, out_file_name):
    train_losses = []

    for epoch_i in range(int(args.num_epoch)):
        logger.info(f"Epoch:[{epoch_i+1}/{args.num_epoch}]")

        if args.add_extra_proj:
            out_dir_name = os.path.join(out_dir + "/matched_names", f"{out_file_name}.json")
        else:
            out_dir_name = os.path.join(out_dir + "/matched_names", f"{out_file_name}.json")
        
        train_loss, matched_names_size_list = train_incre_split_sum_epoch(model, text_model, face_model, sum_loss, frag_loss, train_dataloader,
                                unique_name_dict, unique_name_avg_face_dict, null_face, optimizer, scheduler)
        print(
            "epoch:{}, train_loss:{}".format(
                epoch_i, train_loss
            )
        )

        train_losses.append(train_loss)
        
        matched_name_dict = {}
        for i in range(len(matched_names_size_list)):
            matched_name_dict[i+1] = {}
            matched_name_dict[i+1] = matched_names_size_list[i]

        with open(out_dir_name, "w") as f:
            json.dump(matched_name_dict, f)

    return train_losses


if __name__ == "__main__":
    seed = args.seed

    if seed != 684331:
        if os.path.isdir(os.path.join(args.out_dir[:-12], str(seed))):
            out_dir = os.path.join(args.out_dir[:-12], str(seed))
        else:
            os.mkdir(os.path.join(args.out_dir[:-12], str(seed)))
            out_dir = os.path.join(args.out_dir[:-12], str(seed))    
        print(out_dir)
    else:
        out_dir = args.out_dir

    # DEVICE = torch.device("cuda", args.local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    print(torch.cuda.device_count())
    set_random_seed(seed)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_dir = args.base_dir
    dict_dir = args.dict_dir
    # out_dir = os.path.join(sys_dir, args.out_dir_name)

    if args.split_sum :
        if args.add_extra_proj:
            out_file_name = "stage1:agree{}_alpha{}_".format(args.agree_type, args.alpha) + "bsz{}-{}_F-{}_beta{}_split{}_{}_add-sample{}_{}_add-to{}_make_new{}-{}_face_con{}-{}-{}-{}_add_d_one-{}-{}_{}_nullface-{}".format(
                            args.train_batch_size,
                            args.num_epoch,
                            args.freeze_stage1,
                            args.beta_incre,
                            args.split_sum,
                            args.nomax,
                            args.add_sample,
                            args.noname_to_match,
                            args.add_to,
                            args.make_new,
                            args.combined_direction,
                            args.face_contras,
                            args.face_con_direction,
                            args.only_face_con,
                            args.face_con_replace_diag,
                            args.add_d_one,
                            args.match_proto_agree,
                            args.data_name,
                            args.add_nullface)
        else:
            out_file_name = "stage1:agree{}_alpha{}_".format(args.agree_type, args.alpha) + "bsz{}-{}_split{}_{}_add-sample{}_{}_add-to{}_make_new{}-{}_face_con{}-{}-{}-{}_add_d_one-{}-{}_{}_nullface-{}".format(args.train_batch_size,
                             args.num_epoch,
                             args.split_sum,
                             args.nomax,
                             args.add_sample,
                             args.noname_to_match,
                             args.add_to,
                             args.make_new,
                             args.combined_direction,
                             args.face_contras,
                             args.face_con_direction,
                             args.only_face_con,
                             args.face_con_replace_diag,
                             args.add_d_one,
                             args.match_proto_agree,
                             args.data_name,
                             args.add_nullface)
    else:
        out_file_name = datetime.datetime.now()

    logger = get_logger(os.path.join(out_dir, out_file_name+".log"))
    logger.info(f"GPU:{args.gpu_ids}")
    
    logger.info(f"split sum:{args.split_sum}")
    logger.info(f"add_sample type:{args.add_sample} | add_to:{args.add_to} list")
    logger.info(f"nomax:{args.nomax}")
    logger.info(f"stage1 model:{args.stage1_model_name}")
    # print("batch rest:{}".format(args.batch_rest))
    logger.info(f"add extra projector:{args.add_extra_proj} | freeze stage1:{args.freeze_stage1} | manual add one:{args.manual_add_one}")
    logger.info(f"sample type:{args.sample_type}")
    print("beta incremental:{}".format(args.beta_incre))

    logger.info(f"experiment type:{args.data_type} | text model type:{args.text_model_type}")
    logger.info(f"proj_type:{args.proj_type} | add_bias:{args.add_bias}")
    logger.info(f"optimizer:{args.optimizer_type}")
    logger.info(f"loss_type:{args.loss_type} | direction:{args.direction} | alpha:{args.alpha}")

    logger.info(f"make_new:{args.make_new} | combined_direction:{args.combined_direction} | face_contras:{args.face_contras} | face_con_direction:{args.face_con_direction} | only_face_con:{args.only_face_con} | replace_diag?:{args.face_con_replace_diag} | cal agreement loss for matched prototype faces:{args.match_proto_agree}")

    logger.info(f"add one-one subset into training:{args.add_d_one}")
    logger.info(f"Do no add NONAME to matched list:{args.noname_to_match}")

    logger.info(f"add null face to matched/added faces{args.add_nullface}")

    null_face = torch.randn((1,512)).cuda()

    tokenizer = BertTokenizer.from_pretrained(args.text_model)
    facenet = InceptionResnetV1(pretrained=args.face_model).eval()
    if args.text_model_type == "bert-uncased" or args.text_model_type == "bert-cased" or args.text_model_type == "ernie":
        text_model = BertModel.from_pretrained(args.text_model, output_hidden_states=True)
    else:
        # text_model = CharacterBertModel.from_pretrained(args.charbert_dir)
        print("Please add CharacterBERT to models directory")
    # indexer = CharacterIndexer() # uncomment this line if you want to work with CharacterBERT
    indexer = {} # comment this line if you want to work with CharacterBERT
    special_token_dict = {"additional_special_tokens": args.special_token_list}

    if args.no_stage1:
        unsup_frag_stage1 = UnsupFragAlign(args.ner_dim,
                                           args.proj_type,
                                           args.add_bias,
                                           args.n_features,
                                           args.proj_dim)
    else:
        unsup_frag_stage1 = torch.load(os.path.join(args.stage1_model_dir, args.stage1_model_name))

    if args.add_extra_proj:
        unsup_frag_net = UnsupIncre(unsup_frag_stage1,
                                    args.beta_incre,
                                    args.ner_dim,
                                    args.freeze_stage1,
                                    args.proj_type,
                                    args.add_bias,
                                    args.n_features,
                                    args.proj_dim)
    else:
        unsup_frag_net = unsup_frag_stage1

    if torch.cuda.device_count() > 1:
        unsup_frag_net = nn.DataParallel(unsup_frag_net)
        unsup_frag_net = unsup_frag_net.cuda()
    else:
        unsup_frag_net = unsup_frag_net.cuda()

    loss_type = args.loss_type

    if loss_type == "batch":
        frag_loss = BatchSoftmax(alpha=args.alpha, direction=args.direction, margin=args.margin, agree_type=args.agree_type, max_type=args.max_type)
    elif loss_type == "batch_split":
        frag_loss = BatchSoftmaxSplit(alpha=args.alpha, direction=args.direction)
    else:
        frag_loss = FragAlignLoss(args.pos_factor, args.neg_factor, DEVICE)
    global_loss = GlobalRankLoss(args.beta, args.delta, args.smooth_term, DEVICE)

    if args.add_d_one:
        train_dict_name = os.path.join(dict_dir, args.full_dict_name)
    else:
        train_dict_name = os.path.join(dict_dir, args.dict_name)

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
                             dict_name=train_dict_name)
    face_data = Subset(face_data, range(int(args.data_percent * len(face_data))))

    train_size = int(len(face_data) * 0.7)
    val_size = int(len(face_data) * 0.2)
    test_size = len(face_data) - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        face_data, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed))

    if args.data_type == "test":
        print("train size:{}".format(len(train_set)))
        print("val size:{}".format(len(val_set)))
        print("test size:{}".format(len(test_set)))
    else:
        print("data size:{}".format(len(face_data)))

    zero_pad = ZeroPadCollator()

    if args.split_sum:
        all_loader = DataLoader(face_data, shuffle=args.shuffle, batch_size=args.train_batch_size, collate_fn=zero_pad.collate_fn, num_workers=4)
    elif args.manual_add_one:
        all_loader = DataLoader(face_data, shuffle=args.shuffle, batch_size=args.batch_rest, collate_fn=zero_pad.collate_fn, num_workers=4)
    else:
        all_loader = DataLoader(face_data, shuffle=args.shuffle, batch_size=args.train_batch_size, collate_fn=zero_pad.collate_fn, num_workers=4)

    face_data_full = CelebDataset(base_dir,
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
                                  dict_name=os.path.join(dict_dir, args.full_dict_name))

    all_loader_test = DataLoader(face_data_full, batch_size=args.test_batch_size, num_workers=4)

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, collate_fn=zero_pad.collate_fn, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, collate_fn=zero_pad.collate_fn, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, num_workers=4)

    torch.cuda.current_device()

    if args.data_type == "no_train":
        model, optimizer, scheduler = prep_for_training(unsup_frag_net, args.optimizer_type, len(face_data))
    elif args.split_sum:
        model, optimizer, scheduler = prep_for_training(unsup_frag_net, args.optimizer_type, train_size)
        # if args.add_extra_proj:
        torch.save(model, os.path.join(out_dir, out_file_name + ".pt"))

        sum_loss = BatchSoftmaxNomax()

        with open(os.path.join(dict_dir, args.unique_name_avg_face_dict_name)) as f:
            unique_name_avg_face_dict = json.load(f)
        
        if args.add_sample == "mid-sim":
            with open(os.path.join(dict_dir, "celeb_allname_unique_middle_sim_new.json")) as f:
                unique_name_dict = json.load(f)
        elif args.add_sample == "mid_pdist":
            with open(os.path.join(dict_dir, "celeb_allname_unique_middle_pdist_new.json")) as f:
                unique_name_dict = json.load(f)
        else:
            with open(os.path.join(dict_dir, args.unique_name_dict_name)) as f:
                unique_name_dict = json.load(f)
        
        train_losses = train_all_incre_split_sum(model,
                                                text_model,
                                                facenet, 
                                                sum_loss, frag_loss,
                                                all_loader,
                                                unique_name_dict, unique_name_avg_face_dict,
                                                null_face,
                                                optimizer, scheduler,
                                                out_dir,
                                                out_file_name)

        # if args.add_extra_proj:
        torch.save(model, os.path.join(out_dir, out_file_name + ".pt"))
       
    elif args.sample_type == "True":
        face_data_one = CelebDataset(base_dir,
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
                                     dict_name=args.one_dict_name)
        model, optimizer, scheduler = prep_for_training(unsup_frag_net, args.optimizer_type, len(face_data))

        train_losses = train_all_incre(model,
                                       loss_type,
                                       frag_loss,
                                       global_loss,
                                       all_loader,
                                       optimizer,
                                       scheduler,
                                       face_data_one,
                                       args.train_batch_size)

        if args.add_extra_proj:
            torch.save(model,
                       os.path.join(out_dir, "stage1:agree{}_alpha{}_".format(args.agree_type, args.alpha) + "{}_{}_{}_epoch{}_freeze{}_beta{}.pt".format(args.sample_type,
                                                                                                                     args.train_batch_size,
                                                                                                                        args.batch_rest,
                                                                                                                     args.num_epoch,
                                                                                                                     args.freeze_stage1,
                                                                                                                     args.beta_incre)))
        else:
            torch.save(model,
                       os.path.join(out_dir, "stage1:agree{}_alpha{}_".format(args.agree_type, args.alpha) + "{}_{}_{}_epoch{}.pt".format(args.sample_type,
                                                                                                     args.train_batch_size,
                                                                                                        args.batch_rest,
                                                                                                     args.num_epoch)))
    elif args.sample_type != "True":
        model, optimizer, scheduler = prep_for_training(unsup_frag_net, args.optimizer_type, len(face_data))
        with open(os.path.join(dict_dir, args.unique_name_avg_face_dict_name)) as f:
            unique_name_avg_face_dict = json.load(f)
        with open(os.path.join(dict_dir, args.unique_name_dict_name)) as f:
            unique_name_dict = json.load(f)
        train_losses = train_all_control_one_incre(model,
                                                   loss_type, frag_loss, global_loss,
                                                   all_loader,
                                                   unique_name_avg_face_dict,
                                                   unique_name_dict,
                                                   text_model.to("cuda"), facenet.to("cuda"),
                                                   args.sample_type,
                                                   optimizer, scheduler)

        if args.add_extra_proj:
            torch.save(model,
                       os.path.join(out_dir, "stage1:agree{}_alpha{}_".format(args.agree_type, args.alpha) + "{}_{}_{}_epoch{}_freeze{}_beta{}.pt".format(
                           args.sample_type,
                           args.train_batch_size,
                           args.batch_rest,
                           args.num_epoch,
                           args.freeze_stage1,
                           args.beta_incre)))
        else:
            torch.save(model,
                       os.path.join(out_dir, "stage1:agree{}_alpha{}_".format(args.agree_type, args.alpha) + "{}_{}_{}_epoch{}.pt".format(args.sample_type, args.train_batch_size, args.batch_rest, args.num_epoch)))


    else:
        # To be cleaned
        model, optimizer, scheduler = prep_for_training(unsup_frag_net, args.optimizer_type, len(face_data))

        train_losses = train_all(model,
                                 loss_type,
                                 frag_loss,
                                 global_loss,
                                 all_loader,
                                 optimizer,
                                 scheduler)
        if args.add_extra_proj:
            torch.save(model,
                       os.path.join(out_dir, "stage1:agree{}_alpha{}_".format(args.agree_type, args.alpha) + "stage2_epoch{}_freeze{}_beta{}_split{}.pt".format(args.num_epoch, args.freeze_stage1, args.beta_incre, args.split_sum)))
        else:
            torch.save(model,
                       os.path.join(out_dir, "stage1:agree{}_alpha{}_".format(args.agree_type, args.alpha) + "stage2_epoch{}.pt".format(args.num_epoch)))

    unsup_align_out = {}

    with torch.no_grad():
        test_relu = nn.ReLU()

        if args.data_type == "test":
            test_loader_final = test_loader
        else:
            test_loader_final = all_loader_test

        for idx, data in tqdm(enumerate(test_loader_final)):
            image_name, all_faces, ner_pos_i, caption_raw, ner_list, gt_ner, gt_link, names, ner_ids = data["image_name"][0], data["face_emb"], data["ner_features"], data["caption_raw"],  data["ner_list"], data["gt_ner"], data["gt_link"], data["names"], data["ner_ids"]
            
            ner_context_pos_i = data["ner_context_features"]

            num_face_i = all_faces.size()[2]
            face_list_all = []
            for j in range(num_face_i):
                face_j_list = []  # list for face j in image

                if args.add_extra_proj:

                    face_z_i = (1 - args.beta_incre) * model.projector(all_faces.squeeze(0).squeeze(0)[j].cuda()) \
                               + args.beta_incre * model.model_one_stage.projector(all_faces.squeeze(0).squeeze(0)[j].cuda())

                    if face_z_i.dim() < 1:
                        face_z_i = face_z_i.unsqueeze(0)

                    if args.add_context is True:
                        ner_i = ner_context_pos_i
                    else:
                        ner_i = ner_pos_i

                    ner_z_j = (1 - args.beta_incre) * model.ner_proj(ner_i.squeeze(0).squeeze(0).to(DEVICE)) + args.beta_incre * model.model_one_stage.ner_proj(ner_i.squeeze(0).squeeze(0).to(DEVICE))

                    if args.proj_type == "one":
                        ner_z_all = (1 - args.beta_incre) * model.projector(ner_z_j) + args.beta_incre * model.model_one_stage.projector(ner_z_j)
                    else:
                        ner_z_all = (1 - args.beta_incre) * model.ner_projector(ner_z_j) + args.beta_incre * model.model_one_stage.ner_projector(ner_z_j)
                else:
                    face_z_i = model.projector(all_faces.squeeze(0).squeeze(0)[j].cuda())
                    if face_z_i.dim() < 1:
                        face_z_i = face_z_i.unsqueeze(0)

                    if args.add_context is True:
                        ner_i = ner_context_pos_i
                    else:
                        ner_i = ner_pos_i

                    if args.proj_type == "one":
                        ner_z_all = model.projector(model.ner_proj(ner_i.squeeze(0).squeeze(0).to(DEVICE)))
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
        if args.no_stage1:
            out_dir_name = os.path.join(out_dir,
                                        "mix-train_{}_{}_{}-proj_dim:{}_bias{}_{}data:{}_loss:{}-{}-{}-{}-{}_bsz:{}_shuffle-{}_epoch{}_op:{}_lr{}_noname{}_{}_textModel{}_finetune-{}_mean-{}-{}-layerS{}_{}_batch-rest{}_split{}.json".format(
                                            args.model_name,
                                            args.dict_name[-10:-5],
                                            args.proj_type,
                                            args.proj_dim,
                                            args.add_bias,
                                            args.data_percent,
                                            args.data_type,
                                            args.loss_type,
                                            args.alpha,
                                            args.direction,
                                            args.max_type,
                                            args.agree_type,
                                            args.train_batch_size,
                                            args.shuffle,
                                            args.num_epoch,
                                            args.optimizer_type,
                                            args.lr,
                                            str(args.add_noname),
                                            str(args.cons_noname),
                                            args.text_model_type,
                                            args.fine_tune,
                                            args.use_mean,
                                            args.add_special_token,
                                            args.layer_start,
                                            args.sample_type,
                                            args.batch_rest,
                                            args.split_sum
                                        ))
        elif args.split_sum:
            out_dir_name = os.path.join(out_dir, os.path.join(out_dir, out_file_name + ".json"))

        elif args.manual_add_one:
            if args.add_extra_proj:
                out_dir_name = os.path.join(out_dir, os.path.join(out_dir, "stage1:agree{}_alpha{}_".format(args.agree_type, args.alpha) + "{}_{}_{}_epoch{}_freeze{}_beta{}.json".format(
                    args.sample_type,
                    args.train_batch_size,
                    args.num_epoch,
                    args.batch_rest,
                    args.freeze_stage1,
                    args.beta_incre
                    )))
            else:
                out_dir_name = os.path.join(out_dir,  os.path.join(out_dir, "stage1:agree{}_alpha{}_".format(args.agree_type, args.alpha)+"{}_{}_{}_epoch{}.json".format(args.sample_type,
                                                                                                                                       args.train_batch_size,
                                                                                                                                       args.batch_rest,
                                                                                                                                       args.num_epoch)))
        else:
            if args.add_extra_proj:
                out_dir_name = os.path.join(out_dir,  os.path.join(out_dir, "stage1:agree{}_alpha{}_".format(args.agree_type, args.alpha)+"stage2_epoch{}_freeze{}_beta{}.json".format(args.num_epoch,
                                                                                                                                                     args.freeze_stage1,
                                                                                                                                                     args.beta_incre)))
            else:
                out_dir_name = os.path.join(out_dir,  os.path.join(out_dir, "stage1:agree{}_alpha{}_".format(args.agree_type, args.alpha)+"stage2_epoch{}.json".format(args.num_epoch)))
        with open(out_dir_name, "w") as f:
            json.dump(unsup_align_out, f)

        print(out_dir_name)







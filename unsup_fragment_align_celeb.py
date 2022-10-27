from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scripts.dataset import CelebDataset

import os
from tqdm import tqdm
import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd
from facenet_pytorch import InceptionResnetV1
import torch.distributed as dist
from torch.optim import SGD
from transformers import BertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

from models.unsup_frag import UnsupFragAlign, UnsupFragAlign_FineTune, FragAlignLoss, GlobalRankLoss, BatchSoftmax, BatchSoftmaxSplit
from modules.lars import LARS

# from models.character_bert.modeling.character_bert import CharacterBertModel
# from models.character_bert.utils.character_cnn import CharacterIndexer

import logging

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--base_dir", type=str, default="~/CelebTo/images_ct")
parser.add_argument("--out_dir", type=str, default="~/results/celeb")
parser.add_argument("--dict_name", type=str, default="/CelebrityTo/celeb_dict.json")

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
parser.add_argument("--test_allname", default=False, type=lambda x: (str(x).lower() == 'true'))

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
parser.add_argument("--charbert_dir", type=str, default="/models/character_bert/pretrained-models/general_character_bert")
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


def train_epoch(model, loss_type, frag_loss, global_loss, train_dataloader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        face_emb, ner_features, ner_ids, = batch["face_emb"], batch["ner_features"], batch["ner_ids"]

        if args.fine_tune is True:
            face_j, ner_j = model(face_emb.squeeze(1).to(DEVICE), ner_ids.to(DEVICE))
        else:
            face_j, ner_j = model(face_emb.squeeze(1).cuda(), ner_features.squeeze(1).cuda())

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

        if step % 500 == 0 and not step == 0:
            logger.info(f"step {step} / {len(train_dataloader)} | loss = {tr_loss / nb_tr_steps}")
    return tr_loss / nb_tr_steps


def train_epoch_test_step(model, loss_type, frag_loss, global_loss, train_dataloader, optimizer, scheduler, epoch, out_dir, test_loader_final):
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        if (epoch * len(train_dataloader) + step) % 1000 == 0:
            out_dir_name = os.path.join(out_dir, f"output_{(epoch * len(train_dataloader) + step) // 1000}th_1000steps.json")
            test_by_step(model, epoch, step, test_loader_final, out_dir_name)

        face_emb, ner_features, ner_ids, = batch["face_emb"], batch["ner_features"], batch["ner_ids"]

        if args.fine_tune is True:
            face_j, ner_j = model(face_emb.squeeze(1).to(DEVICE), ner_ids.to(DEVICE))
        else:
            face_j, ner_j = model(face_emb.squeeze(1).cuda(), ner_features.squeeze(1).cuda())

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

        if step % 500 == 0 and not step == 0:
            logger.info(f"step {step} / {len(train_dataloader)} | loss = {tr_loss / nb_tr_steps}")

    return tr_loss / nb_tr_steps


def eval_epoch(model, loss_type, frag_loss, global_loss, val_dataloader):
    model.eval()
    dev_loss = 0
    nb_dev_steps = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_dataloader, desc="Iteration")):
            face_emb, ner_features = batch["face_emb"], batch["ner_features"]

            face_j, ner_j = model(face_emb.squeeze(1).to(DEVICE), ner_features.squeeze(1).to(DEVICE))

            if loss_type == "all":
                a = frag_loss(face_j, ner_j)
                b = global_loss(face_j, ner_j)
                loss = a + b
            else:
                loss = frag_loss(face_j, ner_j)

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


def train(model, loss_type, frag_loss, global_loss, train_dataloader, validation_dataloader, optimizer, scheduler,
):
    train_losses = []
    valid_losses = []

    for epoch_i in range(int(args.num_epoch)):
        train_loss = train_epoch(model, loss_type, frag_loss, global_loss, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, loss_type, frag_loss, global_loss, validation_dataloader)

        logger.info(f"epoch:{epoch_i+1}, train_loss:{train_loss}, valid_loss:{valid_loss}")

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    return train_losses, valid_losses


def train_all(model, loss_type, frag_loss, global_loss, all_dataloader, optimizer, scheduler, out_dir, test_loader_final):
    train_losses = []

    for epoch_i in range(int(args.num_epoch)):
        if args.test_allname:
            train_loss = train_epoch_test_step(model, loss_type, frag_loss, global_loss, all_dataloader, optimizer, scheduler, epoch_i, out_dir, test_loader_final)
        else:
            train_loss = train_epoch(model, loss_type, frag_loss, global_loss, all_dataloader, optimizer, scheduler)

        logger.info(f"Epoch:[{epoch_i+1}/{args.num_epoch}], train_loss:{train_loss}")

        train_losses.append(train_loss)

    return train_losses


class ZeroPadCollator:

    @staticmethod
    def collate_tensors(batch) -> torch.Tensor:
        dims_face_emb = batch[0].dim()
        max_face_size = [max([b.size(i) for b in batch]) for i in range(dims_face_emb)]
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
        caption_raw = self.split_batch(batch, "caption_raw")
        caption_ids = self.split_batch(batch, "caption_ids")
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
        pad_ner_features = self.collate_tensors(ner_features)

        return {
            "num_faces": num_faces,
            "face_tensor": pad_face_tensors,
            "face_emb": pad_face_features,
            "caption_raw": caption_raw,
            "caption_ids": caption_ids,
            "ner_ids": pad_ner_ids,
            "caption_emb": caption_emb,
            "img_rgb": img_rgb,
            "names": names,
            "ner_list": ner_list,
            "ner_features": pad_ner_features,
            "ner_context_features": ner_context_features,
            "word_emb": word_emb,
        }


def test_by_step(model, epoch, step, test_loader_final, out_dir_name):
    unsup_align_out = {}
    
    logger.info(f"Start inference for epoch{epoch}, step{step}")
    with torch.no_grad():
        for idx, data in tqdm(enumerate(test_loader_final)):
            image_name, all_faces, ner_pos_i, caption_raw, ner_list, gt_ner, gt_link, names, ner_ids = data["image_name"][0], data["face_emb"], data["ner_features"], data["caption_raw"], data["ner_list"], data["gt_ner"], data["gt_link"], data["names"], data["ner_ids"]
            
            ner_context_pos_i = data["ner_context_features"]

            num_face_i = all_faces.size()[2]
            face_list_all = []
            for j in range(num_face_i):
                face_j_list = []  # list for face j in image

                face_z_i = model.projector(all_faces.squeeze(0).squeeze(0)[j].cuda())

                if face_z_i.dim() < 1:
                    face_z_i = face_z_i.unsqueeze(0)

                if args.add_context is True:
                    ner_i = ner_context_pos_i
                else:
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

        with open(out_dir_name, "w") as f:
            json.dump(unsup_align_out, f)
    
    logger.info(f"Finish inference for epoch{epoch}, step{step}")


if __name__ == "__main__":
    seed = args.seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    print(torch.cuda.device_count())

    if args.test_allname:
        out_dir = os.path.join(args.out_dir, "test_allname")
    elif seed != 684331:
        os.mkdir(os.path.join(args.out_dir[:-6], str(seed)))
        out_dir = os.path.join(args.out_dir[:-6], str(seed))
        print(out_dir)
    else:
        out_dir = args.out_dir
    out_file_name = "{}_{}_{}-proj_dim:{}_bias{}_{}data:{}_loss:{}-{}-{}-{}-{}_bsz:{}_shuffle-{}_epoch{}_op:{}_lr{}_noname{}_{}_textModel{}_finetune-{}_mean-{}-{}-layerS{}.pt".format(args.model_name,
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
                         args.layer_start,)

    logger = get_logger(os.path.join(out_dir, out_file_name+".log"))
    logger.info(f"GPU:{args.gpu_ids}")
    logger.info(f"experiment type:{args.data_type} | text model type:{args.text_model_type}")
    logger.info(f"Fine-tune BERT:{args.fine_tune}")
    logger.info(f"proj_type:{args.proj_type} | add_bias:{args.add_bias}")
    logger.info(f"optimizer:{args.optimizer_type}")
    logger.info(f"loss_type:{args.loss_type} | direction:{args.direction} | alpha:{args.alpha}")
    logger.info(f"add noname:{args.add_noname} | cons noname:{args.cons_noname}")
    logger.info(f"use mean:{args.use_mean} | layer start:{args.layer_start}")
    logger.info(f"add special token:{args.add_special_token} | margin for hinge loss (if used):{args.margin} | max type:{args.max_type} | use one hot:{args.use_onehot}")
    logger.info(f"test every 1000 steps: {args.test_allname}")

    set_random_seed(seed)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = args.base_dir

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

    if args.fine_tune is True:
        unsup_frag_net = UnsupFragAlign_FineTune(text_model,
                                                 args.ner_dim,
                                                 DEVICE,
                                                 args.fine_tune,
                                                 args.proj_type,
                                                 args.add_bias,
                                                 args.n_features,
                                                 args.proj_dim)
    else:
        unsup_frag_net = UnsupFragAlign(args.ner_dim,
                                        args.proj_type,
                                        args.add_bias,
                                        args.n_features,
                                        args.proj_dim)

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

    all_loader = DataLoader(face_data, shuffle=args.shuffle, batch_size=args.train_batch_size, collate_fn=zero_pad.collate_fn, num_workers=4)
    all_loader_test = DataLoader(face_data, batch_size=args.test_batch_size, num_workers=4)

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, collate_fn=zero_pad.collate_fn, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, collate_fn=zero_pad.collate_fn, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, num_workers=4)

    torch.cuda.current_device()

    if args.data_type == "test":
        model, optimizer, scheduler = prep_for_training(unsup_frag_net, args.optimizer_type, train_size)

        train_losses, valid_losses = train(model,
                                           loss_type,
                                           frag_loss,
                                           global_loss,
                                           train_loader,
                                           val_loader,
                                           optimizer,
                                           scheduler)

    elif args.data_type == "no_train":
        model, optimizer, scheduler = prep_for_training(unsup_frag_net, args.optimizer_type, len(face_data))

    else:
        model, optimizer, scheduler = prep_for_training(unsup_frag_net, args.optimizer_type, len(face_data))

        torch.save(model, os.path.join(out_dir, out_file_name + ".pt"))
        
        logger.info("Start training")
        train_losses = train_all(model,
                                 loss_type,
                                 frag_loss,
                                 global_loss,
                                 all_loader,
                                 optimizer,
                                 scheduler,
                                 out_dir,
                                 all_loader_test)
        logger.info("Finish training")

    torch.save(model, os.path.join(out_dir, out_file_name + ".pt"))

    unsup_align_out = {}

    out_dir_name = os.path.join(out_dir, out_file_name+".json")

    logger.info("Start inference")
    with torch.no_grad():
        test_relu = nn.ReLU()

        if args.data_type == "test":
            test_loader_final = test_loader
        else:
            test_loader_final = all_loader_test

        for idx, data in tqdm(enumerate(test_loader_final)):
            image_name, all_faces, ner_pos_i, caption_raw, ner_list, gt_ner, gt_link, names, ner_ids = data["image_name"][0], data["face_emb"], data["ner_features"], data["caption_raw"], data["ner_list"], data["gt_ner"], data["gt_link"], data["names"], data["ner_ids"]
            
            
            ner_context_pos_i = data["ner_context_features"]

            num_face_i = all_faces.size()[2]
            face_list_all = []
            for j in range(num_face_i):
                face_j_list = []  # list for face j in image

                face_z_i = model.projector(all_faces.squeeze(0).squeeze(0)[j].cuda())

                if face_z_i.dim() < 1:
                    face_z_i = face_z_i.unsqueeze(0)

                if args.add_context is True:
                    ner_i = ner_context_pos_i
                else:
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

        with open(out_dir_name, "w") as f:
            json.dump(unsup_align_out, f)
    
    logger.info("Finish inference")
    print(out_dir_name)







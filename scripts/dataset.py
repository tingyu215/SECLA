import json
import os
# from types import NoneType
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.nn.functional import interpolate

from torchvision import transforms
import re
import cv2
import numpy as np
import nltk
from tqdm import tqdm
from collections import Counter

from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from facenet_pytorch import InceptionResnetV1


class CelebDataset(Dataset):
    def __init__(self, data_dir, tokenizer, indexer, special_token_dict, DEVICE, face_model, text_model, text_model_type, use_mean, layer_start, layer_end, add_special_token, out_face_size=160, face_feat_size=512, use_name_ner="name", add_noname=True, cons_noname=False, dict_name="/cw/working-rose/tingyu/FaceNaming/CelebrityTo/celeb_dict.json"):
        self.base_dir = data_dir
        self.tokenizer = tokenizer
        self.indexer = indexer
        self.special_token_dict = special_token_dict
        self.face_model = face_model.to(DEVICE)
        self.text_model = text_model.to(DEVICE)
        self.text_model_type = text_model_type

        self.use_mean = use_mean
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.add_special_token = add_special_token

        self.DEVICE = DEVICE
        self.out_face_size = out_face_size
        self.face_feat_size = face_feat_size
        self.use_name_ner = use_name_ner
        self.add_noname = add_noname
        self.cons_noname = cons_noname

        with open(dict_name) as f:
            self.data_dict = json.load(f)

    def _prepare_face(self, base_dir, sub_dir, DEVICE, face_bbox, out_face_size, face_feat_size):
        """get face tensors based on bounding box"""
        img_dir = os.path.join(base_dir, sub_dir)
        img = Image.open(img_dir).convert("RGB")
        num_faces = len(face_bbox)
        face_tensors = torch.empty((num_faces, 3, out_face_size, out_face_size))
        face_features = torch.empty((num_faces, face_feat_size))
        for i in range(num_faces):
            crop_face = crop_resize(img, face_bbox[i], out_face_size)
            face_tensors[i] = transforms.ToTensor()(crop_face)
            face_features[i] = self.face_model(face_tensors[i].unsqueeze_(0).to(DEVICE))
        return img_dir, num_faces, face_tensors, face_features

    def gen_ner_emb(self, tokenizer, text_model, ner_list):
        len_ner = len(ner_list)
        ner_features = torch.empty(len_ner, 768)
        ner_ids = []
        for i in range(len_ner):
            ner = ner_list[i]
            if ner.startswith("NONAME"):

                if self.cons_noname:
                    encoded_ids = tokenizer.encode_plus(text="[UNK]",
                                                        add_special_tokens=True,
                                                        return_tensors="pt")["input_ids"]
                else:
                    encoded_ids = tokenizer.encode_plus(text="NONAME",
                                                        add_special_tokens=True,
                                                        return_tensors="pt")["input_ids"]

                ner_id = encoded_ids.to(self.DEVICE)
            else:
                encoded_ids = tokenizer.encode_plus(text=ner,
                                                    add_special_tokens=True,
                                                    return_tensors="pt")["input_ids"]

                ner_id = encoded_ids.to(self.DEVICE)
                
            ner_emb = text_model(ner_id)["pooler_output"]
            ner_features[i] = ner_emb.squeeze()
            ner_ids.append(ner_id)
        ner_ids_padded = self.pad_ner_ids(ner_ids)
        
        return ner_features, ner_ids_padded

    def gen_ner_emb_by_layer(self, tokenizer, text_model, ner_list, add_special_tokens, layer_start, layer_end):
        len_ner = len(ner_list)
        ner_features = torch.empty(len_ner, 768)
        ner_ids = []
        for i in range(len_ner):
            ner = ner_list[i]
            if ner.startswith("NONAME"):
                if self.cons_noname:
                    encoded_ids = tokenizer.encode_plus(text="[UNK]",
                                                        add_special_tokens=add_special_tokens,
                                                        return_tensors="pt")["input_ids"]
                else:
                    encoded_ids = tokenizer.encode_plus(text="NONAME",
                                                        add_special_tokens=add_special_tokens,
                                                        return_tensors="pt")["input_ids"]

                ner_id = encoded_ids.to(self.DEVICE)
            else:
                if len(ner) == 0:
                    encoded_ids = tokenizer.encode_plus(text="[UNK]",
                                                        add_special_tokens=add_special_tokens,
                                                        return_tensors="pt")["input_ids"]

                else:
                    encoded_ids = tokenizer.encode_plus(text=ner,
                                                        add_special_tokens=add_special_tokens,
                                                        return_tensors="pt")["input_ids"]

                ner_id = encoded_ids.to(self.DEVICE)
            if add_special_tokens:
                ner_emb = torch.mean(sum(text_model(ner_id)["hidden_states"][layer_start:layer_end])[:, 1:-1, :], dim=1)
            else:
                ner_emb = torch.mean(sum(text_model(ner_id)["hidden_states"][layer_start:layer_end]), dim=1)
            ner_features[i] = ner_emb.squeeze()
            ner_ids.append(ner_id)
        ner_ids_padded = self.pad_ner_ids(ner_ids)
        return ner_features, ner_ids_padded

    def pad_ner_ids(self, ner_ids_list):
        ner_ids_size_list = [ner_id.size()[1] for ner_id in ner_ids_list]
        max_len = max(ner_ids_size_list)
        padded_ner_ids_list = torch.empty(len(ner_ids_list), max_len)
        
        for i in range(len(ner_ids_list)):
            num_padded = max_len - ner_ids_list[i].size()[1]
            pad_tensor = torch.tensor([self.tokenizer.convert_tokens_to_ids("[PAD]")]*num_padded)

            padded_ner_ids_list[i] = torch.cat((ner_ids_list[i].squeeze(0), pad_tensor.to(self.DEVICE)), 0)

        padded_ner_ids_list = padded_ner_ids_list.type(torch.LongTensor)
        return padded_ner_ids_list.to(self.DEVICE)

    def gen_ner_emb_char(self, tokenizer, indexer, text_model, ner_list):
        len_ner = len(ner_list)
        ner_features = torch.empty(len_ner, 768)
        for i in range(len_ner):
            ner = ner_list[i]
            if ner.startswith("NONAME"):

                if self.cons_noname:
                    x = tokenizer.basic_tokenizer.tokenize("[UNK]")
                    encoded_ids = indexer.as_padded_tensor(x)
                else:
                    x = tokenizer.basic_tokenizer.tokenize("NONAME")
                    encoded_ids = indexer.as_padded_tensor(x)
                ner_ids = encoded_ids.to(self.DEVICE)
            else:
                x = tokenizer.basic_tokenizer.tokenize(ner)
                encoded_ids = indexer.as_padded_tensor(x)
                ner_ids = encoded_ids.to(self.DEVICE)

            ner_emb = text_model(ner_ids)["pooler_output"]
            avg_ner_emb = torch.mean(ner_emb, dim=0)
            ner_features[i] = avg_ner_emb
        return ner_features

    def gen_ner_emb_wiki(self, text_model, ner_list):
        len_ner = len(ner_list)
        ner_features = torch.empty(len_ner, 500)
        for i in range(len_ner):
            ner = ner_list[i]
            if ner.startswith("NONAME"):
                if self.cons_noname:
                    ner_emb = text_model.get_entity_vector("[UNK]")
                else:
                    ner_emb = text_model.get_entity_vector("NONAME")
            else:
                ner_emb = text_model.get_entity_vector(ner)
            ner_features[i] = ner_emb
        return ner_features

    def gen_ner_emb_noname(self, tokenizer, text_model, ner_list):
        if ["NONAME"] in ner_list or ["NONAMEWRONG"] in ner_list:
            ner_features, ner_ids = self.gen_ner_emb(tokenizer, text_model, ner_list)
        else:
            # if no NONAME, add one to the last position of ner_list
            ner_list.append(["NONAME"])
            ner_features, ner_ids = self.gen_ner_emb(tokenizer, text_model, ner_list)
        return ner_features, ner_ids

    def gen_ner_emb_by_layer_noname(self, tokenizer, text_model, ner_list, add_special_token,
                                    layer_start, layer_end):
        if "NONAME" in ner_list or "NONAMEWRONG" in ner_list:
            ner_features, ner_ids = self.gen_ner_emb_by_layer(tokenizer, text_model, ner_list, add_special_token,
                                    layer_start, layer_end)
        else:
            # if no NONAME, add one to the last position of ner_list
            ner_list.append("NONAME")
            ner_features, ner_ids = self.gen_ner_emb_by_layer(tokenizer, text_model, ner_list, add_special_token,
                                    layer_start, layer_end)
        return ner_features, ner_ids

    def gen_ner_emb_char_noname(self, tokenizer, indexer, text_model, ner_list):
        if "NONAME" in ner_list or "NONAMEWRONG" in ner_list:
            ner_features = self.gen_ner_emb_char(tokenizer, indexer, text_model, ner_list)
        else:
            # if no NONAME, add one to the last position of ner_list
            ner_list.append("NONAME")
            ner_features = self.gen_ner_emb_char(tokenizer, indexer, text_model, ner_list)
        return ner_features

    def gen_ner_emb_wiki_noname(self, text_model, ner_list):
        if "NONAME" in ner_list or "NONAMEWRONG" in ner_list:
            ner_features = self.gen_ner_emb_wiki(text_model, ner_list)
        else:
            # if no NONAME, add one to the last position of ner_list
            ner_list.append("NONAME")
            ner_features = self.gen_ner_emb_wiki(text_model, ner_list)
        return ner_features

    def __getitem__(self, index):
        dict_keys = [*self.data_dict]
        key = dict_keys[index]

        img_dir, num_faces, face_tensors, face_features = self._prepare_face(self.base_dir, self.data_dict[key]["img_dir"], self.DEVICE, self.data_dict[key]["bbox"], self.out_face_size, self.face_feat_size)

        face_features = face_features.to(self.DEVICE)
        name_list = self.data_dict[key]["name_list"]
        gt_ner = self.data_dict[key]["name_list"]
        
        if self.add_noname:
            if self.use_mean:
                ner_features, ner_ids = self.gen_ner_emb_by_layer_noname(self.tokenizer, self.text_model, name_list, self.add_special_token, self.layer_start, self.layer_end)

            elif self.text_model_type == "bert-uncased" or self.text_model_type == "bert-cased" or self.text_model_type == "ernie":
                ner_features, ner_ids = self.gen_ner_emb_noname(self.tokenizer, self.text_model, name_list)
            elif self.text_model_type == "charbert":
                ner_features, ner_ids = self.gen_ner_emb_char_noname(self.tokenizer, self.indexer, self.text_model, name_list)
            else:
                ner_features, ner_ids = self.gen_ner_emb_wiki_noname(self.text_model, name_list)
        else:
            ner_features, ner_ids = self.gen_ner_emb(self.tokenizer, self.text_model, name_list)

        # only have gt_link in gt_dict_cleaned_phi_face_name.json now
        gt_link = self.data_dict[key]["gt_link"]

        return{
            "image_name": img_dir,
            "num_faces": num_faces,
            "face_tensor": face_tensors.detach().cpu(),
            "face_emb": face_features.unsqueeze(0).detach().cpu(),
            "caption_raw": {},
            "caption_ids": {},
            "ner_ids": ner_ids.detach().cpu(),
            "caption_emb": {},
            "img_rgb": {},
            "names": name_list,
            "gt_ner": gt_ner,
            "ner_list": {},
            "ner_features": ner_features.unsqueeze(0).detach().cpu(),
            "ner_context_features": {},
            "gt_link": gt_link,
            "word_emb": {},
        }

    def __len__(self):
        return len(self.data_dict)


class FaceDataset(Dataset):
    def __init__(self, base_dir, tokenizer, indexer, special_token_dict, DEVICE, face_model, text_model, text_model_type,
                 use_mean, layer_start, layer_end, add_special_token,
                 out_face_size=160, face_feat_size=512, use_name_ner="ner", add_noname=True, cons_noname=False, dict_name="gt_dict_cleaned.json", no_facenet=False):
        self.base_dir = base_dir
        self.tokenizer = tokenizer
        self.indexer = indexer
        self.special_token_dict = special_token_dict
        self.face_model = face_model.to(DEVICE)
        self.text_model = text_model.to(DEVICE)
        self.text_model_type = text_model_type

        self.use_mean = use_mean
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.add_special_token = add_special_token

        self.DEVICE = DEVICE
        self.out_face_size = out_face_size
        self.face_feat_size = face_feat_size
        self.use_name_ner = use_name_ner
        self.add_noname = add_noname
        self.cons_noname = cons_noname
        self.no_facenet = no_facenet

        with open(os.path.join(base_dir, dict_name)) as f:
            self.data_dict = json.load(f)

        _, unique_words_dict = count_unique_word_names(self.data_dict)
        self.data_dict = make_one_hot_emb(self.data_dict, unique_words_dict, emb_dim=768)

    def _prepare_face(self, base_dir, data_dict, key, DEVICE, out_face_size, face_feat_size):
        img_name = data_dict[key]["img_name"][0]
        img_dir = os.path.join(base_dir, "data/BergData/pics", img_name)

        face_bbox = []
        for i in range(len(data_dict[key]["face_x"])):
            if data_dict[key]["face_x"][i] != -1:
                face_x = data_dict[key]["face_x"][i]
                face_y = data_dict[key]["face_y"][i]
                face_size = data_dict[key]["face_size"][i]

                face_bbox.append([face_x - face_size, face_y - face_size, face_x + face_size, face_y + face_size])
            else:
                continue

        num_faces = len(face_bbox)

        img = Image.open(img_dir).convert("RGB")

        face_tensors = torch.empty((num_faces, 3, out_face_size, out_face_size))
        face_features = torch.empty((num_faces, face_feat_size))

        for i in range(num_faces):
            crop_face = crop_resize(img, face_bbox[i], out_face_size)  # size 3*160*160
            face_tensors[i] = transforms.ToTensor()(crop_face).to(DEVICE)
            if self.no_facenet:
                face_features[i] = self.face_model(face_tensors[i].unsqueeze_(0).to(DEVICE)).squeeze(-1).squeeze(-1)
            else:
                face_features[i] = self.face_model(face_tensors[i].unsqueeze_(0).to(DEVICE))

        return img_name, num_faces, face_tensors, face_features

    def gen_caption_emb(self, tokenizer, indexer, text_model, caption_dir):
        caption_raw = prepare_berg_text(caption_dir)

        encoded_tokens = tokenizer.encode_plus(text=caption_raw,
                                               add_special_tokens=True,
                                               return_tensors="pt")
        caption_ids = encoded_tokens["input_ids"]
        caption_ids = caption_ids.to(self.DEVICE)
        if self.text_model_type == "bert-uncased" or self.text_model_type == "bert-cased" or self.text_model_type == "ernie":
            caption_emb = text_model(caption_ids)["last_hidden_state"]
        else:
            x = tokenizer.basic_tokenizer.tokenize(caption_raw)
            encoded_ids = indexer.as_padded_tensor(x)
            encoded_ids = encoded_ids.to(self.DEVICE)

            caption_emb = text_model(encoded_ids)["pooler_output"]

        return caption_raw, caption_ids, caption_emb

    def gen_ner_emb(self, tokenizer, text_model, ner_list):
        len_ner = len(ner_list)
        ner_features = torch.empty(len_ner, 768)
        ner_ids = []
        for i in range(len_ner):
            ner = ner_list[i][0]
            if ner.startswith("NONAME"):

                if self.cons_noname:
                    encoded_ids = tokenizer.encode_plus(text="[UNK]",
                                                        add_special_tokens=True,
                                                        return_tensors="pt")["input_ids"]
                else:
                    encoded_ids = tokenizer.encode_plus(text="NONAME",
                                                        add_special_tokens=True,
                                                        return_tensors="pt")["input_ids"]

                ner_id = encoded_ids.to(self.DEVICE)
            else:
                encoded_ids = tokenizer.encode_plus(text=ner,
                                                    add_special_tokens=True,
                                                    return_tensors="pt")["input_ids"]

                ner_id = encoded_ids.to(self.DEVICE)
            ner_emb = text_model(ner_id)["pooler_output"]
            ner_features[i] = ner_emb.squeeze()
            ner_ids.append(ner_id)
        ner_ids_padded = self.pad_ner_ids(ner_ids)
        return ner_features, ner_ids_padded

    def gen_ner_emb_by_layer(self, tokenizer, text_model, ner_list, add_special_tokens, layer_start, layer_end):
        len_ner = len(ner_list)
        ner_features = torch.empty(len_ner, 768)
        ner_ids = []
        for i in range(len_ner):
            ner = ner_list[i][0]
            if ner.startswith("NONAME"):
                if self.cons_noname:
                    encoded_ids = tokenizer.encode_plus(text="[UNK]",
                                                        add_special_tokens=add_special_tokens,
                                                        return_tensors="pt")["input_ids"]
                else:
                    encoded_ids = tokenizer.encode_plus(text="NONAME",
                                                        add_special_tokens=add_special_tokens,
                                                        return_tensors="pt")["input_ids"]

                ner_id = encoded_ids.to(self.DEVICE)
            else:
                if len(ner) == 0:
                    encoded_ids = tokenizer.encode_plus(text="[UNK]",
                                                        add_special_tokens=add_special_tokens,
                                                        return_tensors="pt")["input_ids"]

                else:
                    encoded_ids = tokenizer.encode_plus(text=ner,
                                                        add_special_tokens=add_special_tokens,
                                                        return_tensors="pt")["input_ids"]

                ner_id = encoded_ids.to(self.DEVICE)
            if add_special_tokens:
                ner_emb = torch.mean(sum(text_model(ner_id)["hidden_states"][layer_start:layer_end])[:, 1:-1, :], dim=1)
            else:
                ner_emb = torch.mean(sum(text_model(ner_id)["hidden_states"][layer_start:layer_end]), dim=1)
            ner_features[i] = ner_emb.squeeze()
            ner_ids.append(ner_id)
        ner_ids_padded = self.pad_ner_ids(ner_ids)
        return ner_features, ner_ids_padded

    def pad_ner_ids(self, ner_ids_list):
        ner_ids_size_list = [ner_id.size()[1] for ner_id in ner_ids_list]
        max_len = max(ner_ids_size_list)
        padded_ner_ids_list = torch.empty(len(ner_ids_list), max_len)

        for i in range(len(ner_ids_list)):
            num_padded = max_len - ner_ids_list[i].size()[1]
            pad_tensor = torch.tensor([self.tokenizer.convert_tokens_to_ids("[PAD]")]*num_padded)

            padded_ner_ids_list[i] = torch.cat((ner_ids_list[i].squeeze(0), pad_tensor.to(self.DEVICE)), 0)

        padded_ner_ids_list = padded_ner_ids_list.type(torch.LongTensor)
        return padded_ner_ids_list.to(self.DEVICE)

    def gen_ner_emb_char(self, tokenizer, indexer, text_model, ner_list):
        len_ner = len(ner_list)
        ner_features = torch.empty(len_ner, 768)
        for i in range(len_ner):
            ner = ner_list[i][0]
            if ner.startswith("NONAME"):

                if self.cons_noname:
                    x = tokenizer.basic_tokenizer.tokenize("[UNK]")
                    encoded_ids = indexer.as_padded_tensor(x)
                else:
                    x = tokenizer.basic_tokenizer.tokenize("NONAME")
                    encoded_ids = indexer.as_padded_tensor(x)
                ner_ids = encoded_ids.to(self.DEVICE)
            else:
                x = tokenizer.basic_tokenizer.tokenize(ner)
                encoded_ids = indexer.as_padded_tensor(x)
                ner_ids = encoded_ids.to(self.DEVICE)

            ner_emb = text_model(ner_ids)["pooler_output"]
            avg_ner_emb = torch.mean(ner_emb, dim=0)
            ner_features[i] = avg_ner_emb
        return ner_features

    def gen_ner_emb_wiki(self, text_model, ner_list):
        len_ner = len(ner_list)
        ner_features = torch.empty(len_ner, 500)
        for i in range(len_ner):
            ner = ner_list[i][0]
            if ner.startswith("NONAME"):
                if self.cons_noname:
                    ner_emb = text_model.get_entity_vector("[UNK]")
                else:
                    ner_emb = text_model.get_entity_vector("NONAME")
            else:
                ner_emb = text_model.get_entity_vector(ner)
            ner_features[i] = ner_emb
        return ner_features

    def gen_ner_emb_noname(self, tokenizer, text_model, ner_list):
        if ["NONAME"] in ner_list or ["NONAMEWRONG"] in ner_list:
            ner_features, ner_ids = self.gen_ner_emb(tokenizer, text_model, ner_list)
        else:
            # if no NONAME, add one to the last position of ner_list
            ner_list.append(["NONAME"])
            ner_features, ner_ids = self.gen_ner_emb(tokenizer, text_model, ner_list)
        return ner_features, ner_ids

    def gen_ner_emb_by_layer_noname(self, tokenizer, text_model, ner_list, add_special_token,
                                    layer_start, layer_end):
        if ["NONAME"] in ner_list or ["NONAMEWRONG"] in ner_list:
            ner_features, ner_ids = self.gen_ner_emb_by_layer(tokenizer, text_model, ner_list, add_special_token,
                                    layer_start, layer_end)
        else:
            # if no NONAME, add one to the last position of ner_list
            ner_list.append(["NONAME"])
            ner_features, ner_ids = self.gen_ner_emb_by_layer(tokenizer, text_model, ner_list, add_special_token,
                                    layer_start, layer_end)
        return ner_features, ner_ids

    def gen_ner_emb_char_noname(self, tokenizer, indexer, text_model, ner_list):
        if ["NONAME"] in ner_list or ["NONAMEWRONG"] in ner_list:
            ner_features = self.gen_ner_emb_char(tokenizer, indexer, text_model, ner_list)
        else:
            # if no NONAME, add one to the last position of ner_list
            ner_list.append(["NONAME"])
            ner_features = self.gen_ner_emb_char(tokenizer, indexer, text_model, ner_list)
        return ner_features

    def gen_ner_emb_wiki_noname(self, text_model, ner_list):
        if ["NONAME"] in ner_list or ["NONAMEWRONG"] in ner_list:
            ner_features = self.gen_ner_emb_wiki(text_model, ner_list)
        else:
            # if no NONAME, add one to the last position of ner_list
            ner_list.append(["NONAME"])
            ner_features = self.gen_ner_emb_wiki(text_model, ner_list)
        return ner_features

    @staticmethod
    def add_tokens(tokenizer, special_token_dict):
        return tokenizer

    @staticmethod
    def locate_add_tokens(caption_raw, ner_list):
        caption_list = []
        for ner in ner_list:
            if re.search(ner, repr(caption_raw)) is not None:
                start_pos = re.search(ner, repr(caption_raw)).span()[0]  # repr(): for matching non-English characters
                end_pos = re.search(ner, repr(caption_raw)).span()[1]
                caption_list.append(caption_raw[:start_pos] +
                                    "[MASK]" +
                                    caption_raw[end_pos:])
            else:
                caption_list.append(caption_raw)  # unmatched ners --> put special tokens to start & end of the sentence

        return caption_list

    def get_ner_context_emb(self, tokenizer, text_model, ner_list, caption_raw, special_token_dict):
        caption_list = self.locate_add_tokens(caption_raw, ner_list)
        tokenizer = self.add_tokens(tokenizer, special_token_dict)
        ner_context_features = self.gen_ner_emb(tokenizer, text_model, caption_list)
        return ner_context_features

    def __getitem__(self, index):
        dict_keys = [*self.data_dict]
        key = dict_keys[index]

        img_name = self.data_dict[key]["img_name"][0]

        image_name, num_faces, face_tensors, face_features = self._prepare_face(self.base_dir, self.data_dict, key, self.DEVICE, self.out_face_size, self.face_feat_size)

        img_dir = os.path.join(self.base_dir, "data/BergData/pics", img_name)

        img = Image.open(img_dir).convert("RGB")
        img_rgb = transforms.ToTensor()(img).unsqueeze_(0)

        caption_dir = os.path.join(self.base_dir,
                                   "data/BergData/captions",
                                   img_name.replace("/big", "")+".txt")

        caption_raw, caption_ids, caption_emb = self.gen_caption_emb(self.tokenizer, self.indexer, self.text_model, caption_dir)

        name_list = self.data_dict[key]["name_list"]
        ner_list = self.data_dict[key]["ner_list"]
        gt_ner = self.data_dict[key]["ner"]

        ner_unique_list = unique_ner(ner_list)
        if self.use_name_ner == "ner":
            if self.use_mean:
                ner_features, ner_ids = self.gen_ner_emb_by_layer(self.tokenizer, self.text_model, ner_unique_list, self.add_special_token, self.layer_start, self.layer_end)

            elif self.text_model_type == "bert-uncased" or self.text_model_type == "bert-cased" or self.text_model_type == "ernie":
                ner_features, ner_ids = self.gen_ner_emb(self.tokenizer, self.text_model, ner_unique_list)
            elif self.text_model_type == "charbert":
                ner_features, ner_ids = self.gen_ner_emb_char(self.tokenizer, self.indexer, self.text_model, ner_unique_list)
            else:
                ner_features, ner_ids = self.gen_ner_emb_wiki(self.text_model, ner_unique_list)
        elif self.add_noname:
            if self.use_mean:
                ner_features, ner_ids = self.gen_ner_emb_by_layer_noname(self.tokenizer, self.text_model, name_list, self.add_special_token, self.layer_start, self.layer_end)

            elif self.text_model_type == "bert-uncased" or self.text_model_type == "bert-cased" or self.text_model_type == "ernie":
                ner_features, ner_ids = self.gen_ner_emb_noname(self.tokenizer, self.text_model, name_list)
            elif self.text_model_type == "charbert":
                ner_features, ner_ids = self.gen_ner_emb_char_noname(self.tokenizer, self.indexer, self.text_model, name_list)
            else:
                ner_features, ner_ids = self.gen_ner_emb_wiki_noname(self.text_model, name_list)
        else:
            ner_features, ner_ids = self.gen_ner_emb(self.tokenizer, self.text_model, name_list)

        if not self.text_model_type == "bert-uncased" or self.text_model_type == "bert-cased" or self.text_model_type == "ernie":
            ner_context_features = torch.randn(768)
        else:
            ner_context_features = torch.randn(768)

        # only have gt_link in gt_dict_cleaned_phi_face_name.json now
        gt_link = self.data_dict[key]["gt_link"]

        word_emb = self.data_dict[key]["word_emb"]

        return{
            "image_name": image_name,
            "num_faces": num_faces,
            "face_tensor": face_tensors.detach().cpu(),
            "face_emb": face_features.unsqueeze(0).detach().cpu(),
            "caption_raw": caption_raw,
            "caption_ids": caption_ids.detach().cpu(),
            "ner_ids": ner_ids.detach().cpu(),
            "caption_emb": caption_emb.detach().cpu(),
            "img_rgb": img_rgb,
            "names": name_list,
            "gt_ner": gt_ner,
            "ner_list": ner_unique_list,
            "ner_features": ner_features.unsqueeze(0).detach().cpu(),
            "ner_context_features": ner_context_features.unsqueeze(0).detach().cpu(),
            "gt_link": gt_link,
            "word_emb": word_emb.detach().cpu(),
        }

    def __len__(self):
        return len(self.data_dict)


def count_unique_word_names(data_dict):
    word_list = []
    for key in tqdm(data_dict.keys()):
        name_list = data_dict[key]["name_list"]
        for name in name_list:
            word_list.append(nltk.word_tokenize(name[0]))
    flatten_list = []
    for names in word_list:
        flatten_list.extend(names)
    unique_words = Counter(flatten_list)

    id = 0
    for key in unique_words.keys():
        unique_words[key] = id
        id += 1

    return word_list, unique_words


def make_one_hot_emb(data_dict, unique_words_dict, emb_dim=768):
    vocab_size = len(unique_words_dict)

    emb_layer = torch.nn.Embedding(vocab_size, emb_dim)
    for key in tqdm(data_dict.keys()):
        name_list = data_dict[key]["name_list"]
        data_dict[key]["one_hot_vecs"] = {}
        data_dict[key]["word_emb"] = {}

        word_list = []
        one_hot_list = []
        word_emb_list = []
        for name in name_list:
            word_list.extend(nltk.word_tokenize(name[0]))
            id_list = []
            for word in word_list:
                id_list.append(unique_words_dict[word])
            one_hot_vec = F.one_hot(torch.tensor(id_list), vocab_size)
            word_emb = emb_layer(torch.tensor(id_list))
            word_emb_sum = torch.sum(word_emb, dim=0, keepdim=True)
            one_hot_list.append(one_hot_vec)
            word_emb_list.append(word_emb_sum)

        data_dict[key]["one_hot_vecs"] = one_hot_list
        data_dict[key]["word_emb"] = torch.stack(word_emb_list).squeeze(1)
    return data_dict


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


def prepare_berg_text(caption_dir):
    with open(caption_dir, encoding="unicode_escape") as f:
        caption = f.read()

    caption = re.sub("<b>", "", caption)
    caption = re.sub("</b>", "", caption)
    caption = re.sub(" +", " ", caption)

    return caption


def unique_ner(ner_list):
    ner_unique_list = []
    ner_id_list = []
    for i in range(len(ner_list)):
        if ner_list[i][1] not in ner_id_list and len(ner_list[i][0]) > 1:
            ner = ner_list[i][0].replace(" ( R", "")
            ner = ner.replace(" ( L", "")
            ner = ner.replace(" ( R )", "")
            ner = ner.replace(" ( L )", "")
            ner = ner.replace(" ( R)", "")
            ner = ner.replace(" ( L)", "")
            ner = ner.replace(" (R )", "")
            ner = ner.replace(" (L )", "")
            ner = ner.replace(" (R)", "")
            ner = ner.replace(" (L)", "")
            ner = ner.replace(" )", "")
            ner = ner.replace(" (", "")
            ner_unique_list.append(ner)
            ner_id_list.append(ner_list[i][1])
        else:
            continue
    return ner_unique_list


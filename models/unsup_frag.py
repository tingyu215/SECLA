from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class UnsupFragAlign(nn.Module):
    def __init__(self, ner_dim, proj_type="one", add_bias=False, n_features=512, proj_dim=512, no_facenet=False):
        super(UnsupFragAlign, self).__init__()

        self.ner_dim = ner_dim
        self.proj_type = proj_type  # proj_type=one: one projector for both; two: separate projector
        self.n_features = n_features
        self.proj_dim = proj_dim

        if self.proj_type == "two5":
            if no_facenet:
                self.projector = nn.Sequential(
                    nn.Linear(self.n_features, self.n_features, bias=False),
                    nn.ReLU(),
                    nn.Linear(self.n_features, proj_dim, bias=add_bias),
                    nn.ReLU(),
                    nn.Linear(proj_dim, proj_dim, bias=add_bias),
                )
            else:
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
        elif self.proj_type == "two7":
            self.projector = nn.Sequential(
                nn.Linear(self.n_features, self.n_features, bias=False),
                nn.ReLU(),
                nn.Linear(self.n_features, self.n_features // 2, bias=add_bias),
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
                nn.Linear(self.n_features // 2, proj_dim, bias=add_bias),
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
        elif self.proj_type == "two5multi":
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
                nn.Linear(proj_dim, proj_dim*3, bias=add_bias),
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
            nn.ReLU(),
        )
        

    def forward(self, enc_face_emb, enc_ner_emb):

        if self.proj_type == "no_face":  # no projector for face features, this case proj_dim=512
            face_z_i = enc_face_emb
        else:
            face_z_i = self.projector(enc_face_emb)

        ner_z_j = self.ner_proj(enc_ner_emb)  # size 1*1*768 --> 1*1*512
        if self.proj_type == "one":
            ner_z_j = self.projector(ner_z_j)
        elif self.proj_type == "two5multi":
            bsz, seq_len, _= ner_z_j.size()
            ner_z_j = self.ner_projector(ner_z_j).reshape((bsz, seq_len, 3, self.proj_dim))
            print(ner_z_j.size())
        else:
            ner_z_j = self.ner_projector(ner_z_j)

        return face_z_i, ner_z_j


class UnsupFragAlign_FineTune(nn.Module):
    def __init__(self, text_model, ner_dim, DEVICE, fine_tune=True, proj_type="one", add_bias=False, n_features=512, proj_dim=512):
        super(UnsupFragAlign_FineTune, self).__init__()

        self.text_model = text_model
        self.ner_dim = ner_dim
        self.DEVICE = DEVICE
        self.proj_type = proj_type  # proj_type=one: one projector for both; two: separate projector
        self.n_features = n_features

        # freeze first 6 layers of BERT/CharacterBERT/ERNIE
        if fine_tune:
           modules = [self.text_model.embeddings, *self.text_model.encoder.layer[:6]]
           for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        else:
            for param in self.text_model.parameters():
                param.requires_grad = False

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
            nn.ReLU(),
        )

    def create_ner_emb(self, ner_ids):
        enc_ner_emb = torch.stack([self.text_model(ner_ids[i])["pooler_output"] for i in range(ner_ids.size(0))])
        return enc_ner_emb

    def forward(self, enc_face_emb, ner_ids):
        if self.proj_type == "no_face":  # no projector for face features, this case proj_dim=512
            face_z_i = enc_face_emb
        else:
            face_z_i = self.projector(enc_face_emb)
        enc_ner_emb = self.create_ner_emb(ner_ids)
        ner_z_j = self.ner_proj(enc_ner_emb.to(self.DEVICE))  # size 1*1*768 --> 1*1*512
        if self.proj_type == "one":
            ner_z_j = self.projector(ner_z_j)
        else:
            ner_z_j = self.ner_projector(ner_z_j)

        return face_z_i, ner_z_j


def extract_nonzero_faces(batch_features):
    batch_size_true = 0
    list_faces = []
    batch_size = batch_features.size()[0]
    for i in range(batch_size):
        list_face = []  # ner feature for one image
        for j in range(batch_features.size()[1]):
            if torch.sum(batch_features[i][j]) != 0:  # extract non zero features (not padded)
                batch_size_true += 1
                list_face.append(batch_features[i][j])
        list_faces.append(list_face)
    return batch_size_true, batch_size, list_faces


class FragAlignLoss(nn.Module):
    def __init__(self, pos_factor, neg_factor, DEVICE):
        super(FragAlignLoss, self).__init__()

        self.pos_factor = pos_factor
        self.neg_factor = neg_factor
        self.DEVICE = DEVICE

        self.relu = nn.ReLU()

    def cal_pos_sim(self, face_i, ner_j_pos, pos_factor):
        sim = self.relu(1 - pos_factor * 1 * torch.matmul(face_i, torch.transpose(ner_j_pos, 0, 1)))
        sim[sim == 1] = 0
        return torch.sum(sim)

    def cal_neg_sim(self, face_i, ner_j_neg, neg_factor):
        sim_all = 0
        for i in range(ner_j_neg.size()[0]):
            sim_i = self.relu(1 - neg_factor * (-1) * torch.matmul(face_i, torch.transpose(ner_j_neg[i], 0, 1)))
            sim_i[sim_i == 1] = 0
            sim_all += torch.sum(sim_i)
        return sim_all

    def forward(self, face_j, ner_j):

        batch_size_true, batch_size, out_features_face = extract_nonzero_faces(face_j)

        pos_sim_all = 0
        neg_sim_all = 0
        for i in range(batch_size):
            for j in range(len(out_features_face[i])):
                face_i_j_proj = out_features_face[i][j].unsqueeze(0)
                ner_i_pos = ner_j[i]  # pos name: y_i_j = 1
                ner_i_neg = ner_j[torch.arange(ner_j.size(0)) != i]  # neg name: y_i_j = -1

                pos_sim = self.cal_pos_sim(face_i_j_proj, ner_i_pos, self.pos_factor)
                neg_sim = self.cal_neg_sim(face_i_j_proj, ner_i_neg, self.neg_factor)

                pos_sim_all += pos_sim
                neg_sim_all += neg_sim

        loss = (pos_sim_all + neg_sim_all) / batch_size

        return loss


class GlobalRankLoss(nn.Module):
    def __init__(self, beta, delta, smooth_term, DEVICE):
        super(GlobalRankLoss, self).__init__()

        self.beta = beta
        self.delta = delta
        self.smooth_term = smooth_term

        self.DEVICE = DEVICE

        self.relu = nn.ReLU()

    def cal_pos_sim(self, face_i, ner_j_pos):
        sim = self.relu(torch.matmul(face_i, torch.transpose(ner_j_pos, 0, 1)))
        return torch.sum(sim)

    def cal_neg_sim(self, face_i, ner_j_neg, smooth_term):
        sim_neg_all = 0
        for i in range(ner_j_neg.size()[0]):
            sim_i = self.relu(torch.matmul(face_i, torch.transpose(ner_j_neg[i], 0, 1)))
            sim_i[sim_i == 1] = 0
            sim_neg_all += torch.sum(sim_i)
        return sim_neg_all / (ner_j_neg.size()[0] + smooth_term)

    def forward(self, face_j, ner_j):

        _, batch_size, out_features_face = extract_nonzero_faces(face_j)

        rank_all = 0

        for i in range(batch_size):
            for j in range(len(out_features_face[i])):
                face_i_j_proj = out_features_face[i][j].unsqueeze(0)
                ner_i_pos = ner_j[i]  # pos name: y_i_j = 1
                ner_i_neg = ner_j[torch.arange(ner_j.size(0)) != i]  # neg name: y_i_j = -1

                pos_sim = self.cal_pos_sim(face_i_j_proj, ner_i_pos)
                neg_sim = self.cal_neg_sim(face_i_j_proj, ner_i_neg, torch.tensor(self.smooth_term))

                rank_all += self.relu(neg_sim - pos_sim + torch.tensor(self.delta))

        loss = torch.tensor(self.beta) * rank_all / batch_size

        return loss


class BatchSoftmax(nn.Module):
    def __init__(self, alpha=0.5, direction="agree", margin=0.2, agree_type="full", max_type="normal"):
        super(BatchSoftmax, self).__init__()
        self.direction = direction
        self.alpha = alpha
        self.margin = margin
        self.agree_type = agree_type
        self.max_type = max_type

    def forward(self, face_j, ner_j):
        if self.direction == "both":
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            loss1 = batch_softmax(face_ner_match, self.max_type)
            loss2 = batch_softmax(ner_face_match, self.max_type)
            loss = loss1 + loss2
        elif self.direction == "face":
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            loss = batch_softmax(face_ner_match, self.max_type)
        elif self.direction == "name":
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            loss = batch_softmax(ner_face_match, self.max_type)
        elif self.direction == "agree":
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            loss1 = batch_softmax(face_ner_match, self.max_type)
            loss2 = batch_softmax(ner_face_match, self.max_type)
            loss3 = batch_agreement(face_ner_match, ner_face_match, agree_type=self.agree_type, max_type=self.max_type)
            
            loss = loss1 + loss2 + self.alpha * loss3
        elif self.direction == "name_agree":
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            loss2 = batch_softmax(ner_face_match, self.max_type)
            loss3 = batch_agreement(face_ner_match, ner_face_match, agree_type=self.agree_type, max_type=self.max_type)
            
            loss = loss2 + self.alpha * loss3
        elif self.direction == "face_agree":
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            loss1 = batch_softmax(face_ner_match, self.max_type)
            loss3 = batch_agreement(face_ner_match, ner_face_match, agree_type=self.agree_type, max_type=self.max_type)
            
            loss = loss1 + self.alpha * loss3
        elif self.direction == "agree_only":
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            loss = batch_agreement(face_ner_match, ner_face_match, agree_type=self.agree_type, max_type=self.max_type)
        elif self.direction == "agree_agreement_k":
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            loss1 = batch_softmax(face_ner_match, self.max_type)
            loss2 = batch_softmax(ner_face_match, self.max_type)
            loss3 = batch_agreement_topk(face_ner_match, ner_face_match, agree_type=self.agree_type, max_type=self.max_type)
            
            loss = loss1 + loss2 + self.alpha * loss3
        elif self.direction == "dence":
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            loss = batch_dence_corr(face_ner_match, ner_face_match, self.max_type)

        elif self.direction == "both-topk":
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            k = get_k(face_ner_match, ner_face_match)
            loss1 = batch_softmax_topk(face_ner_match, k, self.max_type)
            loss2 = batch_softmax_topk(ner_face_match, k, self.max_type)
            loss = loss1 + loss2
        elif self.direction == "agree-topk":
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            k = get_k(face_ner_match, ner_face_match)
            loss1 = batch_softmax_topk(face_ner_match, k, self.max_type)
            loss2 = batch_softmax_topk(ner_face_match, k, self.max_type)
            loss3 = batch_agreement(face_ner_match, ner_face_match, agree_type=self.agree_type, max_type=self.max_type)
            loss = loss1 + loss2 + self.alpha * loss3
        elif self.direction == "dence-topk":
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            loss = batch_dence_corr_topk(face_ner_match, ner_face_match, self.max_type)

        elif self.direction == "hinge":
            face_j_norm = torch.norm(face_j, dim=-1, keepdim=True)
            face_j = face_j / face_j_norm
            ner_j_norm = torch.norm(ner_j, dim=-1, keepdim=True)
            ner_j = ner_j / ner_j_norm

            loss = cal_hinge_loss(face_j, ner_j, self.margin)
        elif self.direction == "hinge_topk":
            face_j_norm = torch.norm(face_j, dim=-1, keepdim=True)
            face_j = face_j / face_j_norm
            ner_j_norm = torch.norm(ner_j, dim=-1, keepdim=True)
            ner_j = ner_j / ner_j_norm

            loss = cal_hinge_loss_topk(face_j, ner_j, self.margin)
        elif self.direction == "agree-hinge":
            face_j_norm = torch.norm(face_j, dim=-1, keepdim=True)
            face_j = face_j / face_j_norm
            ner_j_norm = torch.norm(ner_j, dim=-1, keepdim=True)
            ner_j = ner_j / ner_j_norm

            loss1 = cal_hinge_loss(face_j, ner_j, self.margin)
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            loss2 = batch_softmax(face_ner_match, self.max_type)
            loss3 = batch_softmax(ner_face_match, self.max_type)
            loss4 = batch_agreement(face_ner_match, ner_face_match, agree_type=self.agree_type, max_type=self.max_type)

            loss = loss1 + loss2 + loss3 + self.alpha * loss4
        elif self.direction == "both-hinge":
            face_j_norm = torch.norm(face_j, dim=-1, keepdim=True)
            face_j = face_j / face_j_norm
            ner_j_norm = torch.norm(ner_j, dim=-1, keepdim=True)
            ner_j = ner_j / ner_j_norm

            loss1 = cal_hinge_loss(face_j, ner_j, self.margin)
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            loss2 = batch_softmax(face_ner_match, self.max_type)
            loss3 = batch_softmax(ner_face_match, self.max_type)

            loss = loss1 + loss2 + loss3

        elif self.direction == "face-hinge":
            face_j_norm = torch.norm(face_j, dim=-1, keepdim=True)
            face_j = face_j / face_j_norm
            ner_j_norm = torch.norm(ner_j, dim=-1, keepdim=True)
            ner_j = ner_j / ner_j_norm

            loss1 = cal_hinge_loss(face_j, ner_j, self.margin)
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            loss2 = batch_softmax(face_ner_match, self.max_type)

            loss = loss1 + loss2

        elif self.direction == "name-hinge":
            face_j_norm = torch.norm(face_j, dim=-1, keepdim=True)
            face_j = face_j / face_j_norm
            ner_j_norm = torch.norm(ner_j, dim=-1, keepdim=True)
            ner_j = ner_j / ner_j_norm

            loss1 = cal_hinge_loss(face_j, ner_j, self.margin)
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            loss2 = batch_softmax(ner_face_match, self.max_type)

            loss = loss1 + loss2

        else:
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            loss = batch_softmax(face_ner_match, self.max_type)
        return loss


class BatchSoftmaxSplit(nn.Module):
    def __init__(self, alpha=0.5, direction="both"):
        super(BatchSoftmaxSplit, self).__init__()
        self.alpha = alpha
        self.direction = direction

    def forward(self, face_j, ner_j):
        face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
        if self.direction == "both":
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            loss1 = batch_softmax_split(face_ner_match)
            loss2 = batch_softmax_split(ner_face_match)
            loss = loss1 + loss2
        elif self.direction == "agree":
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            loss1 = batch_softmax(face_ner_match)
            loss2 = batch_softmax(ner_face_match)
            loss3 = batch_agreement(face_ner_match, ner_face_match, self.direction)

            loss = loss1 + loss2 + self.alpha * loss3
        else:
            loss = batch_softmax_split(face_ner_match)
        return loss


def cal_dence_sim(face_ner_match, max_type):
    # phrase_region_match [B, B, span_len, R]: span_len: names, R: faces
    batch_size, _, num_spans, _ = face_ner_match.size()
    # [B, B, span_len]
    if max_type == "normal":
        face_ner_max = face_ner_match.max(-1).values
        # Logits [B, B]
        face_ner_scores = face_ner_max.sum(-1)
        # Normalize scores
        logits = face_ner_scores.div(
            torch.tensor(num_spans, device=face_ner_scores.device).expand(batch_size).unsqueeze(1).expand(
                face_ner_scores.size())
        )
        targets = torch.arange(
            batch_size, device=face_ner_scores.device
        )
    else:
        logits = conditional_maximum_new(face_ner_match)
        targets = torch.arange(batch_size, device=logits.device)

    return targets, logits


def cal_dence_sim_topk(face_ner_match, k, max_type):
    batch_size, _, _, _ = face_ner_match.size()
    if max_type == "normal":
        face_ner_max = face_ner_match.max(-1).values
        face_ner_scores = torch.topk(face_ner_max, k).values.sum(-1)
    else:
        face_ner_scores = conditional_maximum_new(face_ner_match)

    # no need to normalize scores if we are working with top-k
    targets = torch.arange(
        batch_size, device=face_ner_scores.device
    )

    return targets, face_ner_scores


def get_k(face_ner_match, ner_face_match):

    _, _, num_span_face, _ = face_ner_match.size()

    _, _, num_span_ner, _ = ner_face_match.size()

    k = min(num_span_ner, num_span_face)

    return k


def batch_dence_corr(face_ner_match, ner_face_match, max_type):
    targets, face_logits = cal_dence_sim(face_ner_match, max_type)
    _, ner_logits = cal_dence_sim(ner_face_match, max_type)
    return F.cross_entropy(face_logits + ner_logits, targets)


def batch_dence_corr_topk(face_ner_match, ner_face_match, max_type):
    k = get_k(face_ner_match, ner_face_match)
    targets, face_logits = cal_dence_sim_topk(face_ner_match, k, max_type)
    _, ner_logits = cal_dence_sim_topk(ner_face_match, k, max_type)
    return F.cross_entropy(face_logits + ner_logits, targets)


def batch_softmax_topk(face_ner_match, k, max_type):
    targets, face_logits = cal_dence_sim_topk(face_ner_match, k, max_type)
    return F.cross_entropy(face_logits, targets)


def batch_softmax(phrase_region_match, max_type="normal"):
    if max_type == "normal":
        # phrase_region_match [B, B, span_len, R]: span_len: names, R: faces
        batch_size, _, num_spans, _ = phrase_region_match.size()

        # [B, B, span_len]
        phrase_region_max = phrase_region_match.max(-1).values

        # Logits [B, B]
        phrase_region_scores = phrase_region_max.sum(-1)
        # Normalize scores
        logits = phrase_region_scores.div(
            torch.tensor(num_spans, device=phrase_region_scores.device).expand(batch_size).unsqueeze(1).expand(phrase_region_scores.size())
        )
        targets = torch.arange(
            batch_size, device=phrase_region_scores.device
        )
    else:
        batch_size, _, num_spans, _ = phrase_region_match.size()

        logits = conditional_maximum_new(phrase_region_match)

        targets = torch.arange(
            batch_size, device=logits.device
        )

    return F.cross_entropy(logits, targets)


def batch_agreement(phrase_region_match, region_phrase_match, agree_type = "full", max_type="normal"):
    # phrase_region_match [B, B, span_len, R]: span_len: names, R: faces
    batch_size, _, num_spans, _ = phrase_region_match.size()

    # [B, B, span_len]
    if max_type == "normal":
        phrase_region_max = phrase_region_match.max(-1).values
        region_phrase_max = region_phrase_match.max(-1).values
        # Logits [B, B]
        phrase_region_scores = phrase_region_max.sum(-1)
        region_phrase_scores = region_phrase_max.sum(-1)
        # Normalize scores
        logits_pr = phrase_region_scores.div(
            torch.tensor(num_spans, device=phrase_region_scores.device).expand(batch_size).unsqueeze(1).expand(
                phrase_region_scores.size())
        )
        logits_rp = region_phrase_scores.div(
            torch.tensor(num_spans, device=region_phrase_scores.device).expand(batch_size).unsqueeze(1).expand(
                region_phrase_scores.size())
        )
    else:
        logits_pr = conditional_maximum_new(phrase_region_match)
        logits_rp = conditional_maximum_new(region_phrase_match)

    if agree_type == "full":
        loss = F.mse_loss(logits_pr, logits_rp.transpose(0, 1))

    else:
        loss = F.mse_loss(torch.diag(logits_pr), torch.diag(logits_rp))
    return loss


def batch_agreement_topk(phrase_region_match, region_phrase_match, agree_type = "full", max_type="normal"):
    # phrase_region_match [B, B, span_len, R]: span_len: names, R: faces
    batch_size, _, num_spans, _ = phrase_region_match.size()
    _, _, num_spans_b, _ = region_phrase_match.size()

    k = min(num_spans, num_spans_b)

    # [B, B, span_len]
    if max_type == "normal":
        phrase_region_max = phrase_region_match.max(-1).values
        region_phrase_max = region_phrase_match.max(-1).values
        # Logits [B, B]
        phrase_region_scores = torch.topk(phrase_region_max, k=k, dim=-1).values
        logits_pr = phrase_region_scores.sum(-1)
        region_phrase_scores = torch.topk(region_phrase_max, k=k, dim=-1).values
        logits_rp = region_phrase_scores.sum(-1)
    else:
        logits_pr = conditional_maximum_new(phrase_region_match)
        logits_rp = conditional_maximum_new(region_phrase_match)

    if agree_type == "full":
        loss = F.mse_loss(logits_pr, logits_rp.transpose(0, 1))

    else:
        loss = F.mse_loss(torch.diag(logits_pr), torch.diag(logits_rp))

    return loss


def get_cosine_distance(x, y):
  x_norm = torch.norm(x, dim=-1, keepdim=True)
  y_norm = torch.norm(y, dim=-1, keepdim=True)
  x_new = x/x_norm; y_new= y/y_norm
  return torch.matmul(x_new.unsqueeze(1), y_new.permute(0,2,1))


def cal_hinge_loss(face_j, ner_j, margin):
    face_ner_cos = get_cosine_distance(face_j, ner_j)
    ner_face_cos = get_cosine_distance(ner_j, face_j)

    batch_size, _, num_span_face, _ = face_ner_cos.size()
    _, _, num_span_name, _ = ner_face_cos.size()

    sim_scores = torch.sum(torch.max(face_ner_cos, -1).values, 2)

    # each row: diag pos: corresponding pair, other pos: cross_doc similarities
    sim_scores_normalize = sim_scores.div(
        torch.tensor(num_span_face, device=sim_scores.device).expand(batch_size).unsqueeze(1).expand(
            sim_scores.size())
    )

    sim_scores1 = torch.sum(torch.max(ner_face_cos, -1).values, 2)
    sim_scores1_normalize = sim_scores1.div(
        torch.tensor(num_span_name, device=sim_scores.device).expand(batch_size).unsqueeze(1).expand(
            sim_scores1.size())
    )

    dence_corr_scores = sim_scores_normalize + sim_scores1_normalize

    # reshape diagonal elements to sim_matrix size
    sim_scores_corr = torch.diag(dence_corr_scores).expand_as(dence_corr_scores)
    sim_scores_corr1 = torch.diag(dence_corr_scores).expand_as(dence_corr_scores).t()

    # delete diagonal elements from the score matrix
    mask = torch.eye(dence_corr_scores.size(0)) > .5

    diag_mask = Variable(mask).to(face_j.device)

    dence_corr_scores_no_diag = dence_corr_scores.masked_fill_(diag_mask, 0)

    cost_im = (dence_corr_scores_no_diag - sim_scores_corr + margin).clamp(min=0)
    cost_s = (dence_corr_scores_no_diag - sim_scores_corr1 + margin).clamp(min=0)

    loss = torch.max(cost_im, dim=0).values.sum() + torch.max(cost_s, dim=1).values.sum()

    return loss


def cal_hinge_loss_topk(face_j, ner_j, margin):
    face_ner_cos = get_cosine_distance(face_j, ner_j)
    ner_face_cos = get_cosine_distance(ner_j, face_j)

    batch_size, _, num_span_face, _ = face_ner_cos.size()
    _, _, num_span_name, _ = ner_face_cos.size()

    k = min(num_span_face, num_span_name)

    sim_scores = torch.sum(torch.topk(torch.max(face_ner_cos, -1).values, k, dim=-1).values, 2)
    # print(sim_scores)
    # each row: diag pos: corresponding pair, other pos: cross_doc similarities
    sim_scores_normalize = sim_scores.div(
        torch.tensor(num_span_face, device=sim_scores.device).expand(batch_size).unsqueeze(1).expand(
            sim_scores.size())
    )

    sim_scores1 = torch.sum(torch.topk(torch.max(ner_face_cos, -1).values, k, dim=-1).values, 2)
    sim_scores1_normalize = sim_scores1.div(
        torch.tensor(num_span_name, device=sim_scores.device).expand(batch_size).unsqueeze(1).expand(
            sim_scores1.size())
    )

    dence_corr_scores = sim_scores_normalize + sim_scores1_normalize

    # reshape diagonal elements to sim_matrix size
    sim_scores_corr = torch.diag(dence_corr_scores).expand_as(dence_corr_scores)
    sim_scores_corr1 = torch.diag(dence_corr_scores).expand_as(dence_corr_scores).t()

    # delete diagonal elements from the score matrix
    mask = torch.eye(dence_corr_scores.size(0)) > .5

    diag_mask = Variable(mask).to(face_j.device)

    dence_corr_scores_no_diag = dence_corr_scores.masked_fill_(diag_mask, 0)

    cost_im = (dence_corr_scores_no_diag - sim_scores_corr + margin).clamp(min=0)
    cost_s = (dence_corr_scores_no_diag - sim_scores_corr1 + margin).clamp(min=0)

    loss = torch.max(cost_im, dim=0).values.sum() + torch.max(cost_s, dim=1).values.sum()

    return loss


def conditional_maximum_new(m):
    batch_size, _, num_span_a, num_span_b = m.size()
    m = m.reshape((batch_size * batch_size, num_span_a, num_span_b))

    k = min(num_span_a, num_span_b)
    out_matrix = torch.autograd.Variable(0.0001*torch.ones(batch_size*batch_size), requires_grad=True)
    out_matrix = out_matrix.to(m.device)

    for i in range(batch_size*batch_size):
        max_row = torch.max(m[i], dim=0, keepdim=True).values
        max_column = torch.max(m[i], dim=1, keepdim=True).values

        topk_row = torch.topk(max_row, k).values
        topk_column = torch.topk(max_column.t(), k).values

        out_matrix[i] = torch.sum(topk_row + topk_column)
    out_matrix = out_matrix.reshape((batch_size, batch_size))
    return out_matrix


def batch_softmax_split(phrase_region_match):
    # phrase_region_match [B, B, span_len, R]: span_len: names, R: faces
    batch_size, _, num_spans, _ = phrase_region_match.size()

    # [B, B, span_len]
    phrase_region_max = phrase_region_match.max(-1).values

    logits = phrase_region_max.div(
        torch.tensor(num_spans, device=phrase_region_max.device).expand(batch_size).unsqueeze(1).expand(phrase_region_max.size())
    )

    # compute cross entropy on sample level, num of targets: 0~num_spans
    targets = torch.arange(
        num_spans, device=phrase_region_max.device
    )

    # [B, span_len, span_len], each sample in batch --> [span_len, span_len] diag matrix
    logits_all = torch.empty((batch_size, num_spans, num_spans), device=phrase_region_max.device)


    for i in range(batch_size):
        logits_all[i] = torch.diag(logits[i, i]).type(torch.FloatTensor)

    return F.cross_entropy(logits_all, targets.repeat(batch_size, 1))

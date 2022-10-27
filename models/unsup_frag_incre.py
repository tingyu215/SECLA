import torch.nn as nn

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
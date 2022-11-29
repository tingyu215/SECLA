#!/bin/sh
python /FaceNaming/face_align_word_emb.py --sys_dir /FaceNaming \
--out_dir /export/home1/NoCsBack/working/tingyu/face_naming/unsup_frag \
--dict_name gt_dict_cleaned_phi_face_name.json --gpu_ids 1 --num_gpu 1 --seed 684331 \
--fine_tune False --loss_type batch --alpha 0.15 --direction agree --pos_factor 1 --neg_factor 1 \
--delta 0.25 --smooth_term 3 --beta 2 --optimizer_type adam --data_type train --add_context False --data_percent 1 \
--proj_type two5 --proj_dim 128 --n_features 512 --ner_dim 768 \
--use_name_ner name --add_noname True --cons_noname True \
--add_bias True --model_name unsup_frag --lr 3e-4 --lr_scheduler_tmax 32 --sgd_momentum 0.9 \
--weight_decay 0.1 --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-6 \
--num_epoch 150 --train_batch_size 20 --val_batch_size 32 --test_batch_size 1 \
--shuffle True \
--text_model bert-base-uncased --text_model_type bert-uncased --face_model vggface2 \
--charbert_dir /FaceNaming/models/character_bert/pretrained-models/general_character_bert \
--use_mean True --layer_start -4 --add_special_token True --margin 0.2 \
--agree_type full --max_type normal --use_onehot False


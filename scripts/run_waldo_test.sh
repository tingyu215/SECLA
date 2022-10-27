#!/bin/sh
python /FaceNaming/waldo_test.py --sys_dir /FaceNaming \
--experiment_type celeb_noneg \
--base_dir_name Berg --dict_name /FaceNaming/CelebrityTo/celeb_dict.json \
--gpu_ids 2 \
--waldo_dir /OUTPUTDIR/face_naming/celeb \
--waldo_model_name unsup_frag__dict_two5-proj_dim:128_biasTrue_1.0data:train_loss:batch-0.15-hinge-normal-diag_bsz:20_shuffle-True_epoch3_op:adam_lr0.0003_nonameTrue_True_textModelbert-uncased_finetune-False_mean-True-True-layerS-4.pt.pt \
--alpha 0.15 --agree_type diag --data_name allname \
--add_extra_proj False --beta_incre 0.5 \
--data_type celeb_noneg --add_noname True --data_dict /FaceNaming/CelebrityTo/celeb_dict.json \
--text_model_type bert-uncased --charbert_dir /FaceNaming/models/character_bert/pretrained-models/general_character_bert \
--text_model bert-base-uncased --face_model vggface2 \
--use_mean True --layer_start -4 --add_special_token True \
--use_name_ner name --add_noname True --cons_noname True

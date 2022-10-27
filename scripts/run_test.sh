#!/bin/sh
python /cw/liir_code/NoCsBack/tingyu/FaceNaming/scripts/run_test.py --sys_dir /cw/liir_code/NoCsBack/tingyu/FaceNaming \
--base_dir_name Berg \
--out_dir /export/home1/NoCsBack/working/tingyu/face_naming/unsup_frag \
--result_json_name unsup_frag__name_two5-proj_dim:128_biasTrue_1.0data:train_loss:batch-0.15-agree-normal-diag_bsz:20_shuffle-True_epoch60_op:adam_lr0.0003_nonameTrue_True_textModelbert-uncased_finetune-False_mean-True-True-layerS-4-False-True.json \
--data_type noname_noneg --add_noname True --data_dict gt_dict_cleaned_phi_face_name.json

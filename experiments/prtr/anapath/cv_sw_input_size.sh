#! /bin/bash

DATASET="ANAPATH"
ARCH="PRTR"
HP="forest_sw_input_size"
FILE=${DATASET}_${ARCH}_${HP}

#SBATCH --job-name=prtr_ANAPATH_sw_input_size
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=196:00:00
#SBATCH --mem=400G
#SBATCH --output=/home/mass/ifilesets/ULG/s124930/cellcounting/logs/ANAPATH_PRTR_sw_input_size.txt
export PYTHONPATH=${PYTHONPATH}:/home/mass/ifilesets/ULG/s124930/cellcounting/src/
/home/mass/ifilesets/ULG/s124930/miniconda2/envs/cytomine_sklearndev/bin/python \
/home/mass/ifilesets/ULG/s124930/cellcounting/src/softwares/rt_validation/add_and_run_job.py \
--dataset $DATASET \
--cytomine_working_path /home/mass/ifilesets/ULG/s124930/tmp/ \
--model_file /home/mass/ifilesets/ULG/s124930/cellcounting/models/${FILE}.pkl \
--n_jobs 8 \
--pre_transformer edt \
--pre_alpha 3 \
--sw_input_size 24 \
--sw_input_size 8 \
--sw_input_size 16 \
--sw_input_size 32 \
--sw_output_size 1 \
--sw_extr_mode scoremap_constrained \
--sw_extr_score_thres 0.3 \
--sw_extr_ratio 1.0 \
--sw_colorspace "RGB__rgb RGB__Luv RGB__hsv L__normalized L__sobel1 L__gradmagn" \
--forest_method "ET-regr" \
--forest_n_estimators 15 \
--forest_max_features sqrt \
--cv_save_to /home/mass/ifilesets/ULG/s124930/cellcounting/results/${FILE}.csv \
--cv_mode GKF \
--cv_param 2 \
-vvv

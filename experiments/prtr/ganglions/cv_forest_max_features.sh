#! /bin/bash

DATASET="GANGLIONS"
ARCH="PRTR"
HP="forest_max_features"
FILE=${DATASET}_${ARCH}_${HP}

#SBATCH --job-name=prtr_ganglions_max_features
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=196:00:00
#SBATCH --mem=400G
#SBATCH --output=/home/mass/ifilesets/ULG/s124930/cellcounting/logs/GANGLIONS_PRTR_forest_max_features.txt
export PYTHONPATH=${PYTHONPATH}:/home/mass/ifilesets/ULG/s124930/cellcounting/src/
/home/mass/ifilesets/ULG/s124930/miniconda2/envs/cytomine_sklearndev/bin/python \
/home/mass/ifilesets/ULG/s124930/cellcounting/src/softwares/rt_validation/add_and_run_job.py \
--dataset $DATASET \
--cytomine_working_path /home/mass/ifilesets/ULG/s124930/tmp/ \
--model_file /home/mass/ifilesets/ULG/s124930/cellcounting/models/${FILE}.pkl \
--n_jobs 8 \
--pre_transformer edt \
--pre_alpha 4 \
--sw_input_size 24 \
--sw_output_size 1 \
--sw_extr_mode scoremap_constrained \
--sw_extr_score_thres 0.3 \
--sw_extr_ratio 1.0 \
--sw_colorspace "RGB__rgb RGB__Luv RGB__hsv L__normalized L__sobel1 L__gradmagn" \
--forest_method "ET-regr" \
--forest_n_estimators 15 \
--forest_min_samples_split 100 \
--forest_max_features sqrt \
--forest_max_features 1 \
--forest_max_features 1536 \
--cv_save_to /home/mass/ifilesets/ULG/s124930/cellcounting/results/${FILE}.csv \
--cv_mode GKF \
--cv_param 2 \
-vvv

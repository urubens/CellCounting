#! /bin/bash

DATASET="ANAPATH"
ARCH="SRTC"
HP="assessment"
FILE=${DATASET}_${ARCH}_${HP}

#SBATCH --job-name=SRTC_ANAPATH_assessment
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=196:00:00
#SBATCH --mem=400G
#SBATCH --output=/home/mass/ifilesets/ULG/s124930/cellcounting/logs/ANAPATH_SRTC_assessment.txt
export PYTHONPATH=${PYTHONPATH}:/home/mass/ifilesets/ULG/s124930/cellcounting/src/
/home/mass/ifilesets/ULG/s124930/miniconda2/envs/cytomine_sklearndev/bin/python \
/home/mass/ifilesets/ULG/s124930/cellcounting/src/softwares/rt_model_builder/add_and_run_job.py \
--dataset $DATASET \
--cytomine_working_path /home/mass/ifilesets/ULG/s124930/tmp/ \
--model_file /home/mass/ifilesets/ULG/s124930/cellcounting/models/${FILE}.pkl \
--n_jobs 4 \
--sw_input_size 24 \
--sw_output_size 1 \
--sw_extr_mode scoremap_constrained \
--sw_colorspace "RGB__rgb RGB__Luv RGB__hsv L__normalized L__sobel1 L__gradmagn" \
--forest_method "ET-clf" \
--forest_n_estimators 32 \
--forest_min_samples_split 200 \
--forest_max_features sqrt \
--cv_save_to /home/mass/ifilesets/ULG/s124930/cellcounting/results/${FILE}.csv \
--cv_mode GKF \
--cv_param 3 \
-vvv

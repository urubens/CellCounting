#! /bin/bash

DATASET="CRC"
ARCH="DRTR"
HP="assessment"
FILE=${DATASET}_${ARCH}_${HP}

#SBATCH --job-name=DRTR_CRC_assessment
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=196:00:00
#SBATCH --mem=800G
#SBATCH --output=/home/mass/ifilesets/ULG/s124930/cellcounting/logs/CRC_DRTR_assessment.txt
export PYTHONPATH=${PYTHONPATH}:/home/mass/ifilesets/ULG/s124930/cellcounting/src/
/home/mass/ifilesets/ULG/s124930/miniconda2/envs/cytomine_sklearndev/bin/python \
/home/mass/ifilesets/ULG/s124930/cellcounting/src/softwares/rt_validation/add_and_run_job.py \
--dataset $DATASET \
--cytomine_working_path /home/mass/ifilesets/ULG/s124930/tmp/ \
--model_file /home/mass/ifilesets/ULG/s124930/cellcounting/models/${FILE}.pkl \
--n_jobs 2 \
--pre_transformer density \
--sw_input_size 32 \
--sw_output_size 32 \
--sw_extr_mode random \
--sw_extr_npi 10000 \
--sw_colorspace "RGB__rgb RGB__Luv RGB__hsv L__normalized L__sobel1 L__gradmagn" \
--forest_method "ET-regr" \
--forest_n_estimators 32 \
--forest_min_samples_split 2 \
--forest_max_features sqrt \
--cv_save_to /home/mass/ifilesets/ULG/s124930/cellcounting/results/${FILE}.csv \
--cv_mode GKF \
--cv_param 2 \
-vvv


#! /bin/bash

DATASET="GANGLIONS"
ARCH="FCRN-B"
HP="assessment"
FILE=${DATASET}_${ARCH}_${HP}

sh ../../../install_dgx1.sh
export PYTHONPATH=${PYTHONPATH}:../../../src/
unbuffer python ../../../src/softwares/fcrn_validation/add_and_run_job.py \
--dataset $DATASET \
--cnn_architecture $ARCH \
--cytomine_working_path tmp/ \
--model_file ../../../models/${FILE}.hdf5 \
--cnn_epochs 72 \
--cnn_epochs 120 \
--cnn_batch_size 16 \
--cnn_decay 0.0005 \
--cnn_learning_rate 0.0001 \
--sw_extr_npi 500 \
--sw_input_size 128 \
--sw_extr_mode random \
--post_sigma 1.0 \
--post_sigma 4.0 \
--post_sigma 5.0 \
--pre_transformer density \
--cv_save_to ../../../results/${FILE}.csv \
--cv_mode GKF \
--cv_param 2 \
-vvv 2>&1 | tee ../../../logs/${FILE}.txt
#! /bin/bash

DATASET="BMGRAZ"
ARCH="FCRN-B"
HP="epochs"
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
--cnn_epochs 192 \
--cnn_batch_size 32 \
--cnn_learning_rate 0.01 \
--sw_extr_npi 500 \
--sw_input_size 256 \
--sw_extr_mode random \
--pre_transformer density \
--cv_save_to ../../../results/${FILE}.csv \
--cv_mode GKF \
--cv_param 2 \
-vvv 2>&1 | tee ../../../logs/${FILE}.txt
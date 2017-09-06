#! /bin/bash

export PYTHONPATH=${PYTHONPATH}:../src/
python ../src/softwares/fcrn_validation/add_and_run_job.py \
--dataset BMGRAZ \
--cnn_architecture FCRN-test \
--cytomine_working_path tmp/ \
--model_file ../models/fcrna_bmgraz_test.hdf5 \
--cnn_epochs 1 \
--cnn_epochs 2 \
--cnn_batch_size 16 \
--sw_extr_npi 10 \
--cnn_learning_rate 0.01 \
--sw_input_size 256 \
--sw_extr_mode random \
--pre_transformer density \
--cv_save_to ../results/test.csv \
--cv_mode GKF \
--cv_param 2 \
-vvv
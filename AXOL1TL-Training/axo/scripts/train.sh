#!/bin/bash 

/eos/project/c/cms-axol1tl/Pipeline/Environments/Tf2_14_Ray_Model_Trainer/bin/python axo/ray/ray_contrastive_vae.py \
    --ZeroBiasPath /eos/project/c/cms-axol1tl/automation/ntuple/01_06_2025/ZB_embedded.h5 \
    --BSMPath /eos/project/c/cms-axol1tl/automation/ntuple/01_06_2025/BSM_embedded.h5 \
    --ExperimentName "test" \
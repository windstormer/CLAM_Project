# ./create_job.sh -j run_CAM -p ./CamSeg/run_CAM.sh
# ./create_job.sh -j run_Res18_CLAM -p ./CamSeg/run_Res18_CLAM.sh
# ./create_job.sh -j run_Res18_ScoreCAM -p ./CamSeg/run_Res18_ScoreCAM.sh
# ./create_job.sh -j run_Unet_CLAM -p ./CamSeg/run_Unet_CLAM.sh
# ./create_job.sh -j run_Unet_ScoreCAM -p ./CamSeg/run_Unet_ScoreCAM.sh
# ./create_job.sh -j run_DLab_CLAM -p ./CamSeg/run_DLab_CLAM.sh
# ./create_job.sh -j run_DLab_ScoreCAM -p ./CamSeg/run_DLab_ScoreCAM.sh
# ./create_job.sh -j Cnet_Res18 -p ./Classifier/run_Res18.sh
# ./create_job.sh -j Cnet_Unet -p ./Classifier/run_Unet.sh
# ./create_job.sh -j Cnet_DLab -p ./Classifier/run_DLab.sh
./create_job.sh -j Cnet_Res18_scratch -p ./Classifier/run_Res18_scratch.sh
./create_job.sh -j Cnet_Unet_scratch -p ./Classifier/run_Unet_scratch.sh
./create_job.sh -j Cnet_DLab_scratch -p ./Classifier/run_DLab_scratch.sh

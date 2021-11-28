# ./create_job.sh -j SimCLR_Res18 -p ./SimCLR_train/run_Res18.sh
# ./create_job.sh -j SimCLR_Unet -p ./SimCLR_train/run_Unet.sh -g 4
# ./create_job.sh -j SimCLR_DLab -p ./SimCLR_train/run_DLab.sh -g 4

./create_job.sh -j Cnet_Res18 -p ./Classifier/run_Res18.sh
./create_job.sh -j Cnet_Unet -p ./Classifier/run_Unet.sh
./create_job.sh -j Cnet_DLab -p ./Classifier/run_DLab.sh
# ./create_job.sh -j Cnet_Res50 -p ./Classifier/run_Res50.sh
# ./create_job.sh -j Cnet_Res18_scratch -p ./Classifier/run_Res18_scratch.sh
# ./create_job.sh -j Cnet_Unet_scratch -p ./Classifier/run_Unet_scratch.sh
# ./create_job.sh -j Cnet_DLab_scratch -p ./Classifier/run_DLab_scratch.sh


# ./create_job.sh -j run_CAM_t1 -p ./CamSeg/run_CAM.sh
# ./create_job.sh -j run_Res18_CLAM_t1 -p ./CamSeg/run_Res18_CLAM.sh
# ./create_job.sh -j run_Res18_ScoreCAM_t1 -p ./CamSeg/run_Res18_ScoreCAM.sh
# ./create_job.sh -j run_Unet_CLAM_t1 -p ./CamSeg/run_Unet_CLAM.sh
# ./create_job.sh -j run_Unet_ScoreCAM_t1 -p ./CamSeg/run_Unet_ScoreCAM.sh
# ./create_job.sh -j run_DLab_CLAM_t1 -p ./CamSeg/run_DLab_CLAM.sh
# ./create_job.sh -j run_DLab_ScoreCAM_t1 -p ./CamSeg/run_DLab_ScoreCAM.sh

# ./create_job.sh -j run_CAM_t2 -p ./CamSeg/run_CAM_t2.sh
# ./create_job.sh -j run_Res18_CLAM_t2 -p ./CamSeg/run_Res18_CLAM_t2.sh
# ./create_job.sh -j run_Res18_ScoreCAM_t2 -p ./CamSeg/run_Res18_ScoreCAM_t2.sh
# ./create_job.sh -j run_Unet_CLAM_t2 -p ./CamSeg/run_Unet_CLAM_t2.sh
# ./create_job.sh -j run_Unet_ScoreCAM_t2 -p ./CamSeg/run_Unet_ScoreCAM_t2.sh
# ./create_job.sh -j run_DLab_CLAM_t2 -p ./CamSeg/run_DLab_CLAM_t2.sh
# ./create_job.sh -j run_DLab_ScoreCAM_t2 -p ./CamSeg/run_DLab_ScoreCAM_t2.sh

Training code:
python train.py --data /home/mukit/NDDS/Exported_Data/Trial_for_DGX/hand_cautery/ --batchsize 16 --namefile hand_cautery --gpuids 0,1

python train_1gpu.py --data /home/mukit/NDDS/Exported_Data/cautery_hand/train/noBlood_randColor_onnx/ --datatest /home/mukit/NDDS/Exported_Data/cautery_hand/test/noBlood_randColor_onnx/ --batchsize 16 --outf hc1_onnx --namefile hc1_onnx --epochs 60

# Respose
 python train_1gpu_respose.py --data Datasets/noBlood_randColor/train/ --datatest Datasets/noBlood_randColor/test/ --batchsize 32 --outf rp_v1 --namefile rp_v1 --epochs 60 --network ResPose
 
python train_1gpu_respose.py --data ~/Datasets/fat/ --batchsize 32 --outf rp_meat_v1.1 --namefile rp_meat_v1.1 --epochs 60 --network ResPose --object 010_potted_meat_can_16k --gpuids 1

python train.py --network "DOPE_2.2" --outf "doep2.2_meat" --data "/home/mukit/Datasets/fat/single/010_potted_meat_can_16k/" --epoch 60 --featureNet "vgg" --lr 0.0001

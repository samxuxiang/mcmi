CUDA_VISIBLE_DEVICES=0 python test.py --dataroot ../dataset/cat2dog \
 --fid ./dog_fid.npz --name cat2dog_cyclegan \
 --model cycle_gan --phase test --no_dropout --a2b --resize_size 286 --crop_size 256 --netG resnet_9blocks 


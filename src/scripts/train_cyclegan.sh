python train.py --dataroot ../dataset/cat2dog --name cat2dog_cyclegan --model cycle_gan \
--pool_size 50 --no_dropout --num_threads 8 --lambda_identity 0.0 --batch_size 1 --lr 0.0002 --niter 100 \
--niter_decay 100 --lr_decay_iters 50 --gpu_ids 0

import torch
from options.test_options import TestOptions
from dataset import dataset_single
from saver import save_imgs
import os
import inception_utils
from inception import InceptionV3
import numpy as np
import glob
from models import create_model
import modelss
import torchvision


def compute_lpips(imgs, model_alexnet):
  num_samples = len(imgs)
  dists = []
  for idx1 in range(0,num_samples):
    idx2 = idx1+1
    while idx2 < num_samples:
      img0 = imgs[idx1]
      img1 = imgs[idx2]
      lpips_dist = model_alexnet.forward(img0,img1).item()
      dists.append(lpips_dist) 
      idx2 += 1 
  lpips_score = sum(dists) / len(dists)
  return lpips_score


def main():
  # parse options
  opt = TestOptions().parse()
  opt.num_threads = 0   # test code only supports num_threads = 1
  opt.batch_size = 1    # test code only supports batch_size = 1
  opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
  opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
  opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

  # data loader
  print('\n--- load dataset ---')
  #if opt.a2b:
  dataset = dataset_single(opt, 'A')
  #else:
  #dataset = dataset_single(opt, 'B')
  loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8)


  file_name = opt.fid
  data_mu = np.load(file_name)['mu']  
  data_sigma = np.load(file_name)['sigma']

  # Load inception net
  print('\n--- load inception net ---')
  block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
  model_inception = InceptionV3([block_idx]).cuda()
  model_inception.eval() # set to eval mode
  
  ## Initializing the AlexNet model
  model_alexnet = modelss.PerceptualLoss(model='net-lin',net='alex',use_gpu=True)

  transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(64),
        torchvision.transforms.ToTensor()]) 


  # Load pre-trained model
  print('\n--- load model ---')
  model = create_model(opt)
  all_files = np.arange(200, 210, 10).tolist()

  torch.manual_seed(8)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(8)
  test_sample = 1
  
  for epoch in all_files:
    model.load_networks(epoch)
    model.eval()

    # generate images & accumulate inception activation
    pool1 = []
    pool2 = []
    dists = []

    for idx1, img1 in enumerate(loader):
      img = img1.cuda()
      imgs = [img]
      names = ['input']

      with torch.no_grad(): 
        img = model.test_forward(img, a2b=True)
        imgs = [img]
        imgss = []
        # accumulate inception activation for FID1
        assert (torch.max(img)<=1.0 and torch.min(img)>=-1.0)
        pool_val = model_inception(img)[0]   # bs x 2048 x 1 x 1
        assert(pool_val.size(1)==2048 and pool_val.size(2) == 1 and pool_val.size(3)==1)
        pool_val = pool_val.squeeze()
        pool1 += [np.asarray(pool_val.cpu())]
        
        
        for i in range(1):       
           img = model.test_forward(img, a2b=False)
           img = model.test_forward(img, a2b=True)
           imgs += [img]
        
        # accumulate inception activation for FID2
        assert (torch.max(img)<=1.0 and torch.min(img)>=-1.0)
        pool_val = model_inception(img)[0]   # bs x 2048 x 1 x 1
        assert(pool_val.size(1)==2048 and pool_val.size(2) == 1 and pool_val.size(3)==1)
        pool_val = pool_val.squeeze()
        pool2 += [np.asarray(pool_val.cpu())]

        # test lpips on 4 (1+1+2) generated images
        for i in range(2):       
           img = model.test_forward(img, a2b=False)
           img = model.test_forward(img, a2b=True)
           imgs += [img]
        for i in imgs:
            imgss.append(transform(i[0].cpu()).cuda())
        dist = compute_lpips(imgss, model_alexnet)
        dists.append(dist)
    
    # compute fid score
    lpips_score = sum(dists) / len(dists)
    print('LPIPS score for epoch %d is %f ' %(epoch, lpips_score))
    pool1 = np.vstack(pool1)
    mu, sigma = np.mean(pool1, axis=0), np.cov(pool1, rowvar=False)
    FID1 = inception_utils.numpy_calculate_frechet_distance(mu, sigma, data_mu, data_sigma)
    print('FID1 score for epoch %d is %f ' %(epoch, FID1))
    pool2 = np.vstack(pool2)
    mu, sigma = np.mean(pool2, axis=0), np.cov(pool2, rowvar=False)
    FID2 = inception_utils.numpy_calculate_frechet_distance(mu, sigma, data_mu, data_sigma)
    print('FID2 score for epoch %d is %f ' %(epoch, FID2))
    
    

if __name__ == '__main__':
  main()



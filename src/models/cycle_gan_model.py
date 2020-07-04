import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import numpy as np


class CycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'mutual', 'lower']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'fake_BB']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'fake_AA']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.shrinkNet = networks.init_net(networks.ShrinkNet(), 'normal', 0.02, self.gpu_ids)
       
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain: 
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
         
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_M = torch.optim.Adam(self.shrinkNet.parameters(), 
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), 
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_M)


    def set_input(self, input, input_anchors):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.bs = self.real_A.shape[0]
        
        self.anchor_A = input_anchors['A' if AtoB else 'B'].to(self.device)  # cat
        self.anchor_B = input_anchors['B' if AtoB else 'A'].to(self.device)  # dog
        self.anchor_bs = self.anchor_A.shape[0]

        # no need to check if real input image is repeated in anchors
        # ==> random cropping + random flipping
        self.anchor_A[0] = self.real_A
        self.anchor_B[0] = self.real_B
    
        

        


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_BB =  self.netG_A(self.rec_A) # G_A(G_B(G_A(A)))
        self.rec_AA = self.netG_B(self.fake_BB) # G_B(G_A(G_B(G_A(A))))

        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))      
        self.fake_AA =  self.netG_B(self.rec_B) # G_B(G_A(G_B(B)))      
        self.rec_BB = self.netG_A(self.fake_AA) # G_A(G_B(G_A(G_B(B))))      


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D



    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B
        fake_BB = self.fake_BB
        rec_BB = self.rec_BB
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A2 = self.backward_D_basic(self.netD_A, self.real_B, fake_BB)
        self.loss_D_A3 = self.backward_D_basic(self.netD_A, self.real_B, rec_BB)
        (0.33*self.loss_D_A + 0.33*self.loss_D_A2 + 0.33*self.loss_D_A3).backward()
        total_loss = (self.loss_D_A + self.loss_D_A2 + self.loss_D_A3)
        

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A
        fake_AA = self.fake_AA
        rec_AA = self.rec_AA
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B2 = self.backward_D_basic(self.netD_B, self.real_A, fake_AA)
        self.loss_D_B3 = self.backward_D_basic(self.netD_B, self.real_A, rec_AA)
        (0.33*self.loss_D_B + 0.33*self.loss_D_B2 + 0.33*self.loss_D_B3).backward()
        total_loss = (self.loss_D_B + self.loss_D_B2 + self.loss_D_B3)



    def get_lower_bound(self, probs, target):
        same_idx = target.view(-1).nonzero()
        select_same = probs.view(-1)[same_idx]
        joint = select_same + np.log(self.anchor_bs)
        marginal = torch.logsumexp(probs, dim=1)
        lower_bound = torch.mean(joint - marginal)
        return lower_bound

    def get_upper_bound(self, probs, target):
        same_idx = target.view(-1).nonzero()
        diff_idx = (1-target).view(-1).nonzero()
        select_same = probs.view(-1)[same_idx]
        joint = select_same + np.log(self.anchor_bs-1)
        select_diff = probs.view(-1)[diff_idx].reshape(probs.size(0),-1)
        marginal = torch.logsumexp(select_diff, dim=1)
        upper_bound = torch.mean(joint - marginal)
        return upper_bound


    def mutual_loss_func(self, upper_bound, lower_bound):
        margin = -0.2
        mutual_loss = upper_bound - lower_bound
        mutual_loss = torch.nn.functional.relu(mutual_loss + margin)
        return mutual_loss


    def cosine_sim(self, x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1,w2).clamp(min=eps)


    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss (no idt loss on cat2dog)
        self.loss_idt_A = 0
        self.loss_idt_B = 0
        
        ### NCE loss ###
        S = 2.0
        M = 2.0
        target=torch.zeros(self.anchor_bs).to(self.device)
        target[0] = 1.0
        
        shrink_A = self.shrinkNet(self.anchor_A.detach()).reshape(self.anchor_bs,-1)
        shrink_ABA = self.shrinkNet(self.rec_A.detach()).reshape(self.bs,-1)
        shrink_ABAB = self.shrinkNet(self.fake_BB).reshape(self.bs,-1)
        shrink_ABABA = self.shrinkNet(self.rec_AA).reshape(self.bs,-1)
        shrink_B = self.shrinkNet(self.anchor_B.detach()).reshape(self.anchor_bs,-1)
        shrink_BAB = self.shrinkNet(self.rec_B.detach()).reshape(self.bs,-1)
        shrink_BABA = self.shrinkNet(self.fake_AA).reshape(self.bs,-1)
        shrink_BABAB = self.shrinkNet(self.rec_BB).reshape(self.bs,-1)

        ABA_probs = torch.abs(self.cosine_sim(shrink_A,  shrink_ABA).reshape(self.bs, -1)) * S - M 
        ABAB_probs = torch.abs(self.cosine_sim(shrink_A,  shrink_ABAB).reshape(self.bs, -1)) * S - M 
        ABABA_probs = torch.abs(self.cosine_sim(shrink_A,  shrink_ABABA).reshape(self.bs, -1)) * S - M 
        BAB_probs = torch.abs(self.cosine_sim(shrink_B,  shrink_BAB).reshape(self.bs, -1)) * S - M 
        BABA_probs = torch.abs(self.cosine_sim(shrink_B,  shrink_BABA).reshape(self.bs, -1)) * S - M 
        BABAB_probs = torch.abs(self.cosine_sim(shrink_B,  shrink_BABAB).reshape(self.bs, -1)) * S - M 


        lower_bound_ABA = self.get_lower_bound(ABA_probs, target)
        lower_bound_ABAB = self.get_lower_bound(ABAB_probs, target)
        upper_bound_ABAB = self.get_upper_bound(ABAB_probs, target)
        upper_bound_ABABA = self.get_upper_bound(ABABA_probs, target)
        lower_bound_BAB = self.get_lower_bound(BAB_probs, target)
        lower_bound_BABA = self.get_lower_bound(BABA_probs, target)
        upper_bound_BABA = self.get_upper_bound(BABA_probs, target)
        upper_bound_BABAB = self.get_upper_bound(BABAB_probs, target)

        self.loss_mutual = self.mutual_loss_func(upper_bound_ABAB, lower_bound_ABA) + \
                           self.mutual_loss_func(upper_bound_ABABA, lower_bound_ABAB) + \
                           self.mutual_loss_func(upper_bound_BABA, lower_bound_BAB) +\
                           self.mutual_loss_func(upper_bound_BABAB, lower_bound_BABA)
        
    
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_A2 = self.criterionGAN(self.netD_A(self.fake_BB), True)
        self.loss_G_A3 = self.criterionGAN(self.netD_A(self.rec_BB), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_G_B2 = self.criterionGAN(self.netD_B(self.fake_AA), True)
        self.loss_G_B3 = self.criterionGAN(self.netD_B(self.rec_AA), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = 0.33*(self.loss_G_A + self.loss_G_A2 + self.loss_G_A3)  + \
                      0.33*(self.loss_G_B + self.loss_G_B2 + self.loss_G_B3)  + \
                      0.1*self.loss_cycle_A + 0.1*self.loss_cycle_B + 0.5*self.loss_mutual 
        self.loss_G.backward()



    
    def backward_M(self):
        """Calculate the loss for generators G_A and G_B"""
        ### NCE loss ###
        S = 2.0
        M = 2.0
        target=torch.zeros(self.anchor_bs).to(self.device)
        target[0] = 1.0
        
        shrink_A = self.shrinkNet(self.anchor_A.detach()).reshape(self.anchor_bs,-1)
        shrink_ABA = self.shrinkNet(self.rec_A.detach()).reshape(self.bs,-1)
        shrink_ABAB = self.shrinkNet(self.fake_BB.detach()).reshape(self.bs,-1)
        shrink_ABABA = self.shrinkNet(self.rec_AA.detach()).reshape(self.bs,-1)
        shrink_B = self.shrinkNet(self.anchor_B.detach()).reshape(self.anchor_bs,-1)
        shrink_BAB = self.shrinkNet(self.rec_B.detach()).reshape(self.bs,-1)
        shrink_BABA = self.shrinkNet(self.fake_AA.detach()).reshape(self.bs,-1)
        shrink_BABAB = self.shrinkNet(self.rec_BB.detach()).reshape(self.bs,-1)

        ABA_probs = torch.abs(self.cosine_sim(shrink_A,  shrink_ABA).reshape(self.bs, -1)) * S - M 
        ABAB_probs = torch.abs(self.cosine_sim(shrink_A,  shrink_ABAB).reshape(self.bs, -1)) * S - M 
        ABABA_probs = torch.abs(self.cosine_sim(shrink_A,  shrink_ABABA).reshape(self.bs, -1)) * S - M 

        BAB_probs = torch.abs(self.cosine_sim(shrink_B,  shrink_BAB).reshape(self.bs, -1)) * S - M 
        BABA_probs = torch.abs(self.cosine_sim(shrink_B,  shrink_BABA).reshape(self.bs, -1)) * S - M 
        BABAB_probs = torch.abs(self.cosine_sim(shrink_B,  shrink_BABAB).reshape(self.bs, -1)) * S - M 

        lower_bound_ABA = self.get_lower_bound(ABA_probs, target)
        lower_bound_ABAB = self.get_lower_bound(ABAB_probs, target)
        lower_bound_ABABA = self.get_lower_bound(ABABA_probs, target)
        lower_bound_BAB = self.get_lower_bound(BAB_probs, target)
        lower_bound_BABA = self.get_lower_bound(BABA_probs, target)
        lower_bound_BABAB = self.get_lower_bound(BABAB_probs, target)

        self.loss_lower = lower_bound_ABA + lower_bound_ABAB + lower_bound_ABABA +\
                          lower_bound_BAB + lower_bound_BABA + lower_bound_BABAB
        (-self.loss_lower).backward()


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()
        self.optimizer_M.zero_grad()  
        self.backward_M()             
        self.optimizer_M.step()

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights



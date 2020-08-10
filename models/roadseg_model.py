import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import os


class RoadSegModel(BaseModel):
    def name(self):
        return 'RoadSegModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def initialize(self, opt, dataset):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['segmentation']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['rgb_image', 'another_image', 'label', 'output']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['RoadSeg']

        # load/define networks
        self.netRoadSeg = networks.define_RoadSeg(dataset.num_labels, use_sne=opt.use_sne, init_type=opt.init_type, init_gain= opt.init_gain, gpu_ids= self.gpu_ids)
        # define loss functions
        self.criterionSegmentation = networks.SegmantationLoss(class_weights=None).to(self.device)

        if self.isTrain:
            # initialize optimizers
            self.optimizers = []
            self.optimizer_RoadSeg = torch.optim.SGD(self.netRoadSeg.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
            #self.optimizer_RoadSeg = torch.optim.Adam(self.netRoadSeg.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_RoadSeg)
            self.set_requires_grad(self.netRoadSeg, True)

    def set_input(self, input):
        self.rgb_image = input['rgb_image'].to(self.device)
        self.another_image = input['another_image'].to(self.device)
        self.label = input['label'].to(self.device)
        self.image_names = input['path']
        self.image_oriSize = input['oriSize']

    def forward(self):
        self.output = self.netRoadSeg(self.rgb_image, self.another_image)

    def get_loss(self):
        self.loss_segmentation = self.criterionSegmentation(self.output, self.label)

    def backward(self):
        self.loss_segmentation.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_RoadSeg.zero_grad()
        self.get_loss()
        self.backward()
        self.optimizer_RoadSeg.step()

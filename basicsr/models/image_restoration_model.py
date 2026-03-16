import importlib
import os
import torch
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, tensor2img, imwrite

loss_module = importlib.import_module('basicsr.models.losses')


class ImageCleanModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(
                self.net_g,
                load_path,
                self.opt['path'].get('strict_load_g', True),
                param_key=self.opt['path'].get('param_key', 'params')
            )

        if self.is_train:
            self.init_training_settings()

    # ----------------------------------------
    # TRAIN SETTINGS
    # ----------------------------------------
    def init_training_settings(self):

        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use EMA with decay: {self.ema_decay}')

            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            self.model_ema(0)
            self.net_g_ema.eval()

        # losses
        self.cri_pix = None
        if train_opt.get('pixel_opt'):
            pixel_opt = train_opt['pixel_opt'].copy()
            pixel_type = pixel_opt.pop('type')
            self.cri_pix = getattr(loss_module, pixel_type)(**pixel_opt).to(self.device)

        self.cri_percep = None
        if train_opt.get('perceptual_opt'):
            percep_opt = train_opt['perceptual_opt'].copy()
            percep_type = percep_opt.pop('type')
            self.cri_percep = getattr(loss_module, percep_type)(**percep_opt).to(self.device)

        self.cri_tv = None
        if train_opt.get('tv_opt'):
            tv_opt = train_opt['tv_opt'].copy()
            tv_type = tv_opt.pop('type')
            self.cri_tv = getattr(loss_module, tv_type)(**tv_opt).to(self.device)

        self.setup_optimizers()
        self.setup_schedulers()

    # ----------------------------------------
    # OPTIMIZER
    # ----------------------------------------
    def setup_optimizers(self):

        train_opt = self.opt['train']
        optim_params = [v for v in self.net_g.parameters() if v.requires_grad]

        optim_type = train_opt['optim_g'].pop('type')

        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError

        self.optimizers.append(self.optimizer_g)

    # ----------------------------------------
    # DATA
    # ----------------------------------------
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def feed_train_data(self, data):
        self.feed_data(data)

    # ----------------------------------------
    # TRAIN STEP
    # ----------------------------------------
    def optimize_parameters(self, current_iter):

        self.optimizer_g.zero_grad()

        preds = self.net_g(self.lq)

        if isinstance(preds, (list, tuple)):
            preds1, preds2, preds3 = preds
        else:
            preds1 = preds
            preds2 = F.interpolate(preds1, scale_factor=0.5, mode='bilinear', align_corners=False)
            preds3 = F.interpolate(preds1, scale_factor=0.25, mode='bilinear', align_corners=False)

        gt1 = self.gt
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)

        total_loss = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l1 = self.cri_pix(preds1, gt1)
            l2 = self.cri_pix(preds2, gt2)
            l3 = self.cri_pix(preds3, gt3)
            l_pix = l1 + 0.5*l2 + 0.25*l3
            total_loss += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_percep:
            l_percep, _ = self.cri_percep(preds1, gt1)
            total_loss += l_percep
            loss_dict['l_percep'] = l_percep

        if self.cri_tv:
            l_tv = self.cri_tv(preds1)
            total_loss += l_tv
            loss_dict['l_tv'] = l_tv

        total_loss.backward()

        if self.opt['train'].get('use_grad_clip'):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)

        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    # ----------------------------------------
    # TEST
    # ----------------------------------------
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
            if isinstance(self.output, (list, tuple)):
                self.output = self.output[0]
        self.net_g.train()

    # ----------------------------------------
    # VISUALS
    # ----------------------------------------
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    # ----------------------------------------
    # VALIDATION
    # ----------------------------------------
    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img=False, rgb2bgr=True, use_image=True):

        logger = get_root_logger()
        logger.info(f'Start Validation {current_iter}')

        self.net_g.eval()

        for idx, val_data in enumerate(dataloader):

            img_name = os.path.splitext(os.path.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()

            sr_img = tensor2img([visuals['result']])
            gt_img = tensor2img([visuals['gt']])

            if save_img:
                save_path = os.path.join(
                    self.opt['path']['visualization'],
                    f'{img_name}_{current_iter}.png'
                )
                imwrite(sr_img, save_path)

        self.net_g.train()
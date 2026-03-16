import logging
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str


# -------------------------
# SELF ENSEMBLE FUNCTION
# -------------------------
def forward_x8(model, x):

    def _transform(v, op):
        if op == 'v':
            v = v.flip(2)
        elif op == 'h':
            v = v.flip(3)
        elif op == 't':
            v = v.transpose(2, 3)
        return v

    lr_list = [x]

    for tf in ['v', 'h', 't']:
        lr_list.extend([_transform(t, tf) for t in lr_list])

    sr_list = [model.net_g(aug) for aug in lr_list]

    for i in range(len(sr_list)):
        if i > 3:
            sr_list[i] = _transform(sr_list[i], 't')
        if i % 4 > 1:
            sr_list[i] = _transform(sr_list[i], 'h')
        if (i % 4) % 2 == 1:
            sr_list[i] = _transform(sr_list[i], 'v')

    output = torch.stack(sr_list, dim=0).mean(dim=0)

    return output


def main():

    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)

    log_file = osp.join(
        opt['path']['log'],
        f"test_{opt['name']}_{get_time_str()}.log"
    )

    logger = get_root_logger(
        logger_name='basicsr',
        log_level=logging.INFO,
        log_file=log_file
    )

    logger.info(get_env_info())
    logger.info(dict2str(opt))

    test_loaders = []

    for phase, dataset_opt in sorted(opt['datasets'].items()):

        if phase != 'test':
            continue

        test_set = create_dataset(dataset_opt)

        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed']
        )

        logger.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}"
        )

        test_loaders.append(test_loader)

    model = create_model(opt)

    # -------------------------
    # PATCH MODEL FOR X8 ENSEMBLE
    # -------------------------
    original_forward = model.net_g.forward

    def new_forward(x):
        return forward_x8(model, x)

    model.net_g.forward = new_forward

    # -------------------------
    # RUN TESTING
    # -------------------------
    for test_loader in test_loaders:

        test_set_name = test_loader.dataset.opt['name']

        logger.info(f'Testing {test_set_name}...')

        rgb2bgr = opt['val'].get('rgb2bgr', True)
        use_image = opt['val'].get('use_image', True)

        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['val']['save_img'],
            rgb2bgr=rgb2bgr,
            use_image=use_image
        )


if __name__ == '__main__':
    main()
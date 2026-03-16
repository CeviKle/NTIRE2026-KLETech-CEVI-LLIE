import argparse
import datetime
import logging
import time
import torch
import torch.backends.cudnn as cudnn
from os import path as osp
import os
import glob

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (
    MessageLogger,
    get_env_info,
    get_root_logger,
    get_time_str,
    init_tb_logger,
    set_random_seed
)
from basicsr.utils.options import dict2str, parse


# ------------------------------------------------
# CUDA SETTINGS
# ------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ------------------------------------------------
# OPTIONS
# ------------------------------------------------
def parse_options(is_train=True):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt",
        type=str,
        default="options/train.yml",
        help="Path to option YAML file."
    )

    args = parser.parse_args()

    opt = parse(args.opt, is_train=is_train)

    opt["dist"] = False
    opt["rank"], opt["world_size"] = 0, 1

    seed = opt.get("manual_seed", 100)
    set_random_seed(seed)

    if "path" not in opt:
        opt["path"] = {}

    opt["path"]["experiments_root"] = "./experiments"
    opt["path"]["models"] = "./experiments/models"
    opt["path"]["training_states"] = "./experiments/training_states"
    opt["path"]["log"] = "./experiments/log"

    return opt


# ------------------------------------------------
# LOGGER
# ------------------------------------------------
def init_loggers(opt):

    os.makedirs(opt["path"]["log"], exist_ok=True)

    log_file = osp.join(
        opt["path"]["log"],
        f"train_{opt['name']}_{get_time_str()}.log"
    )

    logger = get_root_logger(
        logger_name="basicsr",
        log_level=logging.INFO,
        log_file=log_file
    )

    logger.info(get_env_info())
    logger.info(dict2str(opt))

    tb_logger = None

    if opt.get("logger") and opt["logger"].get("use_tb_logger", False):

        tb_logger = init_tb_logger(
            log_dir=osp.join("tb_logger", opt["name"])
        )

    return logger, tb_logger


# ------------------------------------------------
# DATA
# ------------------------------------------------
def create_train_val_dataloader(opt, logger):

    train_set = create_dataset(opt["datasets"]["train"])

    train_sampler = EnlargedSampler(
        train_set,
        opt["world_size"],
        opt["rank"],
        opt["datasets"]["train"].get("dataset_enlarge_ratio", 1),
    )

    train_loader = create_dataloader(
        train_set,
        opt["datasets"]["train"],
        num_gpu=1,
        dist=False,
        sampler=train_sampler,
        seed=opt.get("manual_seed", 0),
    )

    total_iters = int(opt["train"]["total_iter"])

    logger.info(
        f"\nNumber of train images: {len(train_set)}"
        f"\nBatch size: {opt['datasets']['train']['batch_size_per_gpu']}"
        f"\nTotal iterations: {total_iters}"
    )

    val_loader = None

    if opt.get("val"):

        val_set = create_dataset(opt["datasets"]["val"])

        val_loader = create_dataloader(
            val_set,
            opt["datasets"]["val"],
            num_gpu=1,
            dist=False,
            sampler=None,
            seed=opt.get("manual_seed", 0),
        )

        logger.info(f"Number of val images: {len(val_set)}")

    return train_loader, val_loader, total_iters


# ------------------------------------------------
# MAIN
# ------------------------------------------------
def main():

    opt = parse_options(is_train=True)

    torch.cuda.set_device(0)

    cudnn.benchmark = True
    cudnn.deterministic = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision("high")

    logger, tb_logger = init_loggers(opt)

    train_loader, val_loader, total_iters = create_train_val_dataloader(opt, logger)

    model = create_model(opt)

    # ------------------------------------------------
    # AUTO RESUME
    # ------------------------------------------------
    ckpt_dir = "./forced_checkpoints_LLIE"

    resume_iter = 0

    if os.path.exists(ckpt_dir):

        ckpts = glob.glob(os.path.join(ckpt_dir, "net_g_*.pth"))

        if len(ckpts) > 0:

            latest_ckpt = max(
                ckpts,
                key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0])
            )

            resume_iter = int(
                os.path.basename(latest_ckpt).split("_")[-1].split(".")[0]
            )

            logger.info(f"Loading checkpoint {latest_ckpt}")

            state_dict = torch.load(latest_ckpt, map_location="cuda")

            model.net_g.load_state_dict(state_dict, strict=False)

    current_iter = resume_iter

    epoch = resume_iter // len(train_loader)

    logger.info(f"Resume iteration {resume_iter}")

    # ------------------------------------------------
    # CUDA PREFETCH
    # ------------------------------------------------
    prefetcher = CUDAPrefetcher(train_loader, opt)

    start_time = time.time()

    # ------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------
    while current_iter < total_iters:

        epoch += 1

        train_loader.sampler.set_epoch(epoch)

        prefetcher.reset()

        train_data = prefetcher.next()

        while train_data is not None:

            current_iter += 1

            if current_iter > total_iters:
                break

            model.update_learning_rate(
                current_iter,
                warmup_iter=opt["train"].get("warmup_iter", -1),
            )

            model.feed_train_data(train_data)

            model.optimize_parameters(current_iter)

            # ---------------- LOG ----------------
            if current_iter % opt["logger"]["print_freq"] == 0:

                log_vars = {
                    "epoch": epoch,
                    "iter": current_iter,
                    "lrs": model.get_current_learning_rate(),
                }

                log_vars.update(model.get_current_log())

                MessageLogger(opt, current_iter, tb_logger)(log_vars)

            # ---------------- SAVE ----------------
            if current_iter % opt["logger"]["save_checkpoint_freq"] == 0:

                save_dir = "./forced_checkpoints_LLIE"

                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(
                    save_dir,
                    f"net_g_{current_iter}.pth"
                )

                torch.save(model.net_g.state_dict(), save_path)

                logger.info(f"Saved checkpoint {save_path}")

            # ---------------- VALIDATION ----------------
            if val_loader and current_iter % opt["val"]["val_freq"] == 0:

                model.validation(
                    val_loader,
                    current_iter,
                    tb_logger,
                    opt["val"]["save_img"],
                )

            train_data = prefetcher.next()

    # ------------------------------------------------
    # TRAIN END
    # ------------------------------------------------
    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time))
    )

    logger.info(f"Training finished. Time consumed: {consumed_time}")

    if tb_logger:
        tb_logger.close()


if __name__ == "__main__":
    main()
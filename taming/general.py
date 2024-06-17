# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
import random
import importlib
import pathlib
from typing import Tuple, List, Dict, ClassVar
import numpy as np
from omegaconf import OmegaConf
from datetime import datetime

import torch
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger

from .callback import *

import os
import random
import importlib
import pathlib
from typing import Tuple, List, Dict, ClassVar
import numpy as np
from omegaconf import OmegaConf
from datetime import datetime

import torch
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger

from .callback import *


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_obj_from_str(name: str, reload: bool = False) -> ClassVar:
    module, cls = name.rsplit(".",
                              1)  # 把taming.models.topkgcnvqgan.VQModelvis 分成taming.models.topkgcnvqgan 和 VQModelvis
    # 后面的数字1 就是从后面开始分成几个

    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    return getattr(importlib.import_module(module, package=None), cls)


def initialize_from_config(config: OmegaConf) -> object:
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def setup_callbacks(exp_config: OmegaConf, config: OmegaConf, base_path: str) -> Tuple[List[Callback], WandbLogger]:
    now = datetime.now().strftime('%d%m%Y_%H%M%S')

    basedir = pathlib.Path(base_path, exp_config.name, now)
    os.makedirs(basedir, exist_ok=True)

    setup_callback = SetupCallback(config, exp_config, basedir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=setup_callback.ckptdir,
        filename=exp_config.name + "-{epoch:02d}" + "-{val/rec_loss:.2f}",
        monitor="val/rec_loss",
        # filename=exp_config.name+"-{epoch:02d}"+"-{val/loss:.2f}",
        # monitor="val/loss",
        save_last=True,
        save_top_k=5,
        mode='min',
        every_n_epochs=1,
        verbose=False,
    )
    os.makedirs(setup_callback.logdir / 'wandb', exist_ok=True)
    logger = WandbLogger(save_dir=str(setup_callback.logdir), name=exp_config.name + "_" + str(now), offline=True)
    logger_img_callback = ImageTextLogger(exp_config.batch_frequency, exp_config.max_images)

    return [setup_callback, checkpoint_callback, logger_img_callback], logger


def get_config_from_file(config_file: str) -> Dict:
    config_file = OmegaConf.load(config_file)

    if 'base_config' in config_file.keys():
        if config_file['base_config'] == "default_base":
            base_config = get_default_config()
        elif config_file['base_config'].endswith(".yaml"):
            base_config = get_config_from_file(config_file['base_config'])

        config_file = {key: value for key, value in config_file if key != "base_config"}

        return OmegaConf.merge(base_config, config_file)

    return config_file

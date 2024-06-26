# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import sys
import shutil
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch

os.environ['TORCH_HOME'] = './pretrained_weights'
os.environ['WANDB_API_KEY'] = 'a889988eebcb9e1fbed4d762c81ed0326eff8b12'


from taming.general import get_config_from_file, initialize_from_config, setup_callbacks


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # if os.path.exists(Path('experiments')):
    #     shutil.rmtree(Path('experiments'), ignore_errors=True)
    # if not os.path.exists(Path('experiments')):
    #     os.mkdir(Path('experiments'))
    # print('renew experiments!')
    parser.add_argument('-c', '--config', type=str, default='cub_lgvqgan')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-nn', '--num_nodes', type=int, default=1)
    parser.add_argument('-ng', '--num_gpus', type=int, default=1)
    parser.add_argument('-u', '--update_every', type=int, default=1)
    parser.add_argument('-e', '--epochs', type=int, default=500)
    parser.add_argument('-lr', '--base_lr', type=float, default=4.5e-6)
    parser.add_argument('-a', '--use_amp', default=False, action='store_true')
    parser.add_argument('-b', '--batch_frequency', type=int, default=750)
    parser.add_argument('-m', '--max_images', type=int, default=4)

    parser.add_argument('-sp', '--save_path', type=str, default="./exp/cub_lgvqgan")

    parser.add_argument('-lp', '--load_pretrain', type=str, default="last.ckpt")
    parser.add_argument('-p', '--isload_pretrain', type=bool, default=False)

    args = parser.parse_args()

    # print(list(range(args.num_gpus)))
    set_gpu(','.join([str(i) for i in range(args.num_gpus)]))

    # set_gpu(str(args.num_gpus))

    # Set random seed
    pl.seed_everything(args.seed)

    # Load configuration
    config = get_config_from_file(Path("configs_text") / (args.config + ".yaml"))
    exp_config = OmegaConf.create({"name": args.config, "epochs": args.epochs, "update_every": args.update_every,
                                   "base_lr": args.base_lr, "use_amp": args.use_amp,
                                   "batch_frequency": args.batch_frequency,
                                   "max_images": args.max_images})

    # Build model
    model = initialize_from_config(config.model)
    model.learning_rate = exp_config.update_every * args.num_gpus * config.dataset.params.batch_size * exp_config.base_lr

    # Setup callbacks
    callbacks, logger = setup_callbacks(exp_config, config, args.save_path)
    # Build data modules
    print('setting dataset')
    data = initialize_from_config(config.dataset)
    # data.args = args
    data.prepare_data()
    # data.setup()

    # Build trainer
    print('setting training')

    trainer = pl.Trainer(max_epochs=exp_config.epochs,
                         precision=16 if exp_config.use_amp else 32,
                         callbacks=callbacks,
                         gpus=args.num_gpus,
                         num_nodes=args.num_nodes,
                         strategy="ddp" if args.num_nodes > 1 or args.num_gpus > 1 else None,
                         accumulate_grad_batches=exp_config.update_every,
                         logger=logger)

    # Train
    print('start training')
    if args.isload_pretrain == True:
        if os.path.exists(args.load_pretrain):
            print("load success...")
            trainer.fit(model, data, ckpt_path=args.load_pretrain)

        else:
            print("load failed...")
            trainer.fit(model, data)
    else:
        print("load failed...")
        trainer.fit(model, data)

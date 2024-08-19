import argparse
import os
import signal

from lightning.pytorch.callbacks import BatchSizeFinder

# do this before importing numpy! (doing it right up here in case numpy is dependency of e.g. json)
os.environ["MKL_NUM_THREADS"] = "1"  # noqa: E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa: E402

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment

from config.default import cfg
from lib.datasets.datamodules import DataModuleTraining
from lib.models.MicKey.model import MicKeyTrainingModel
from lib.models.MicKey.modules.utils.training_utils import create_exp_name, create_result_dir
import random
import shutil


def train_model(args):
    cfg.merge_from_file(args.dataset_config)
    cfg.merge_from_file(args.config)

    exp_name = create_exp_name(args.experiment, cfg)
    print('Start training of ' + exp_name)

    cfg.DATASET.SEED = random.randint(0, 1000000)

    model = MicKeyTrainingModel(cfg)

    jobId = os.getenv("SLURM_JOB_ID")
    taskId = os.getenv('SLURM_ARRAY_JOB_ID')
    job_id = int(taskId) if taskId else int(jobId)
    ckpt_dir = os.path.join(args.path_weights, exp_name, str(job_id))

    checkpoint_vcre_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='{epoch}-best_vcre',
        save_top_k=1,
        verbose=True,
        monitor='val_vcre/auc_vcre',
        mode='max'
    )

    checkpoint_pose_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='{epoch}-best_pose',
        save_top_k=1,
        verbose=True,
        monitor='val_AUC_pose/auc_pose',
        mode='max'
    )

    epochend_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='{epoch}-last',
        save_last=True,
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True
    )

    lr_monitoring_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    create_result_dir(os.path.join(ckpt_dir, 'config.yaml'))
    shutil.copyfile(args.config, os.path.join(ckpt_dir, 'config.yaml'))

    config_dictionary = dict(
        yaml=cfg,
    )

    logger = WandbLogger(project="mickey", name=exp_name, id=str(job_id), resume="allow",
                         config=config_dictionary)

    trainer = pl.Trainer(devices=cfg.TRAINING.NUM_GPUS,
                         log_every_n_steps=cfg.TRAINING.LOG_INTERVAL,
                         val_check_interval=cfg.TRAINING.VAL_INTERVAL,
                         limit_val_batches=cfg.TRAINING.VAL_BATCHES,
                         max_epochs=cfg.TRAINING.EPOCHS,
                         logger=logger,
                         callbacks=[checkpoint_pose_callback, lr_monitoring_callback, epochend_callback,
                                    checkpoint_vcre_callback,
                                    # BatchSizeFinder()
                                    ],
                         num_sanity_val_steps=0,
                         gradient_clip_val=cfg.TRAINING.GRAD_CLIP,
                         plugins=SLURMEnvironment(requeue_signal=signal.SIGHUP, auto_requeue=False),
                         enable_progress_bar=False,
                         )

    datamodule_end = DataModuleTraining(cfg)
    print('Training with {:.2f}/{:.2f} image overlap'.format(cfg.DATASET.MIN_OVERLAP_SCORE,
                                                             cfg.DATASET.MAX_OVERLAP_SCORE))

    if args.resume:
        ckpt_path = args.resume
    else:
        ckpt_path = None

    trainer.fit(model, datamodule_end, ckpt_path=ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file', default='config/MicKey/curriculum_learning.yaml')
    parser.add_argument('--dataset_config', help='path to dataset config file', default='config/datasets/mapfree.yaml')
    parser.add_argument('--experiment', help='experiment name', default='MicKey_default')
    parser.add_argument('--path_weights', help='path to the directory to save the weights', default='weights/')
    parser.add_argument('--resume', help='resume from checkpoint path', default=None)
    args = parser.parse_args()
    train_model(args)
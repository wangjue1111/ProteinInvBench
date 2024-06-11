import datetime
import os
import sys
sys.path.append(os.getcwd())
os.environ["WANDB_API_KEY"] = "3afd3131afecd5d6e3eb1a05274f3a67bdbb2b1f"
import warnings
warnings.filterwarnings("ignore")

import argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer import Trainer
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
from model_interface import MInterface
from data_interface import DInterface
from src.tools.logger import SetupCallback,BackupCodeCallback
import math
from shutil import ignore_patterns
import yaml

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--res_dir', default='./train/results', type=str)
    parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--pretrained_path', default='/gaozhangyang/experiments/ProteinInvBench/model_zoom/CATH4.3/BlockGAT/best-epoch=49-recovery=0.537.pth', type=str)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('--dataset', default='CATH4.3') # AF2DB_dataset, CATH_dataset
    parser.add_argument('--model_name', default='KWDesign_BlockGAT', choices=['StructGNN', 'GraphTrans', 'GVP', 'GCA', 'AlphaDesign', 'ESMIF', 'PiFold', 'ProteinMPNN', 'BlockGAT', 'KWDesign', 'KWDesign_BlockGAT'])
    parser.add_argument('--lr', default=4e-4, type=float, help='Learning rate')
    parser.add_argument('--lr_scheduler', default='cosine')
    parser.add_argument('--offline', default=1, type=int)
    parser.add_argument('--seed', default=111, type=int)
    
    # dataset parameters
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pad', default=1024, type=int)
    parser.add_argument('--min_length', default=40, type=int)
    parser.add_argument('--data_root', default='./data/')
    
    # Training parameters
    parser.add_argument('--epoch', default=10, type=int, help='end epoch')
    parser.add_argument('--augment_eps', default=0.0, type=float, help='noise level')
    args = parser.parse_args()
    return args




def load_callbacks(args):
    callbacks = []
    
    logdir = str(os.path.join(args.res_dir, args.ex_name))
    
    ckptdir = os.path.join(logdir, "checkpoints")
    
    callbacks.append(BackupCodeCallback(os.path.dirname(args.res_dir),logdir, ignore_patterns=ignore_patterns('results*', 'pdb*', 'metadata*', 'vq_dataset*')))
    

    metric = "recovery"
    sv_filename = 'best-{epoch:02d}-{recovery:.3f}'
    callbacks.append(plc.ModelCheckpoint(
        monitor=metric,
        filename=sv_filename,
        save_top_k=15,
        mode='max',
        save_last=True,
        dirpath = ckptdir,
        verbose = True,
        every_n_epochs = args.check_val_every_n_epoch,
    ))

    
    now = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")
    cfgdir = os.path.join(logdir, "configs")
    callbacks.append(
        SetupCallback(
                now = now,
                logdir = logdir,
                ckptdir = ckptdir,
                cfgdir = cfgdir,
                config = args.__dict__,
                argv_content = sys.argv + ["gpus: {}".format(torch.cuda.device_count())],)
    )
    
    
    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval=None))
    return callbacks



if __name__ == "__main__":
    args = create_parser()
    print(args)
    pl.seed_everything(args.seed)
    model_config = yaml.load(open('src/models/configs/' + args.model_name + '.yaml'),Loader=yaml.FullLoader)
    
    
    data_module = DInterface(**vars(args))
    data_module.setup()
    
    gpu_count = torch.cuda.device_count()
    args.steps_per_epoch = math.ceil(len(data_module.trainset)/args.batch_size/gpu_count)
    print(f"steps_per_epoch {args.steps_per_epoch},  gpu_count {gpu_count}, batch_size{args.batch_size}")

    model_config.update(vars(args))
    model = MInterface(**model_config)

    
    trainer_config = {
        'gpus': -1,  # Use all available GPUs
        'max_epochs': args.epoch,  # Maximum number of epochs to train for
        'num_nodes': 1,  # Number of nodes to use for distributed training
        "strategy": 'deepspeed_stage_2', # 'ddp', 'deepspeed_stage_2
        "precision": '32', # "bf16", 16
        'accelerator': 'gpu',  # Use distributed data parallel
        'callbacks': load_callbacks(args),
        'logger': plog.WandbLogger(
                    project = 'PiFold',
                    name=args.ex_name,
                    save_dir=str(os.path.join(args.res_dir, args.ex_name)),
                    offline = args.offline,
                    id = "_".join(args.ex_name.split("/")),
                     entity = "wangjue4396"),
        'gradient_clip_val':1.0
    }

    trainer_opt = argparse.Namespace(**trainer_config)
    trainer = Trainer.from_argparse_args(trainer_opt)
    
    # trainer.fit(model, data_module)
    trainer.test(model, data_module, ckpt_path = '/gaozhangyang/experiments/ProteinInvBench/train/results/KWDesign_BlockGAT/checkpoints/best-epoch=09-recovery=0.584.ckpt')
    
    print(trainer_config)

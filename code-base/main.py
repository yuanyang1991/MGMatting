import argparse
import os
from pprint import pprint

import toml
import torch
from torch.utils.data import DataLoader

import utils
from dataloader.data_generator import DataGenerator
from dataloader.data_generator_instance_wise import DataGenerator_Instance_Wise
from dataloader.image_file import ImageFileTrain, ImageFileTest
from dataloader.prefetcher import Prefetcher
from trainer import Trainer
from trainer_instance_wise import Trainer_Instance_Wise
from utils import CONFIG


def get_trainer(train_dataloader, test_dataloader, logger, tb_logger):
    if CONFIG.train.use_instance_wise:
        return Trainer_Instance_Wise(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            logger=logger,
            tb_logger=tb_logger)
    else:
        return Trainer(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            logger=logger,
            tb_logger=tb_logger)


def get_train_data_generator(train_image_file):
    if CONFIG.train.use_instance_wise:
        return DataGenerator_Instance_Wise(train_image_file,
                                           phase='train')
    else:
        return DataGenerator(train_image_file, phase='train')


def main():
    # Train or Test
    if CONFIG.phase.lower() == "train":
        # set distributed training
        if CONFIG.dist:
            CONFIG.gpu = CONFIG.local_rank
            torch.cuda.set_device(CONFIG.gpu)
            torch.distributed.init_process_group(backend='gloo', init_method='env://')
            CONFIG.world_size = torch.distributed.get_world_size()

        # Create directories if not exist.
        if CONFIG.local_rank == 0:
            utils.make_dir(CONFIG.log.logging_path)
            utils.make_dir(CONFIG.log.tensorboard_path)
            utils.make_dir(CONFIG.log.checkpoint_path)
        # Create a logger
        logger, tb_logger = utils.get_logger(CONFIG.log.logging_path,
                                             CONFIG.log.tensorboard_path,
                                             logging_level=CONFIG.log.logging_level)
        train_image_file = ImageFileTrain(alpha_dir=CONFIG.data.train_alpha,
                                          fg_dir=CONFIG.data.train_fg,
                                          bg_dir=CONFIG.data.train_bg,
                                          alpha_ext=CONFIG.data.train_alpha_ext,
                                          fg_ext=CONFIG.data.train_fg_ext,
                                          bg_ext=CONFIG.data.train_bg_ext
                                          )
        test_image_file = ImageFileTest(alpha_dir=CONFIG.data.test_alpha,
                                        merged_dir=CONFIG.data.test_merged,
                                        trimap_dir=CONFIG.data.test_trimap,
                                        alpha_ext=CONFIG.data.test_alpha_ext,
                                        merged_ext=CONFIG.data.test_merged_ext,
                                        trimap_ext=CONFIG.data.test_triamp_ext
                                        )

        train_dataset = get_train_data_generator(train_image_file)
        test_dataset = DataGenerator(test_image_file, phase='val')

        if CONFIG.dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            train_sampler = None
            test_sampler = None

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=CONFIG.model.batch_size,
                                      shuffle=(train_sampler is None),
                                      num_workers=CONFIG.data.workers,
                                      pin_memory=True,
                                      sampler=train_sampler,
                                      drop_last=True)
        train_dataloader = Prefetcher(train_dataloader)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=CONFIG.data.workers,
                                     sampler=test_sampler,
                                     drop_last=False)

        trainer = get_trainer(train_dataloader, test_dataloader, logger, tb_logger)
        trainer.train()
    else:
        raise NotImplementedError("Unknown Phase: {}".format(CONFIG.phase))


if __name__ == '__main__':
    print('Torch Version: ', torch.__version__)
    os.environ['RANK'] = '0'  # 根据实际进程修改
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--config', type=str, default='config/train.toml')
    parser.add_argument('--local_rank', type=int, default=0)

    # Parse configuration
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")

    CONFIG.phase = args.phase
    CONFIG.log.logging_path = os.path.join(CONFIG.log.logging_path, CONFIG.version)
    CONFIG.log.tensorboard_path = os.path.join(CONFIG.log.tensorboard_path, CONFIG.version)
    CONFIG.log.checkpoint_path = os.path.join(CONFIG.log.checkpoint_path, CONFIG.version)
    if args.local_rank == 0:
        print('CONFIG: ')
        pprint(CONFIG)
    CONFIG.local_rank = args.local_rank

    # Train
    main()

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
sys.path.append('.')

from training.config import get_config
from training.dataset import MakeupDataset
from training.solver import Solver
from training.utils import create_logger, print_args


def main(config, args):
    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)
    
    dataset = MakeupDataset(config)
    data_loader = DataLoader(dataset, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS, shuffle=True)
    
    solver = Solver(config, args, logger)
    solver.train(data_loader)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='elegant')
    parser.add_argument("--save_path", type=str, default='results', help="path to save model")
    parser.add_argument("--load_folder", type=str, help="path to load model", 
                        default=None)
    parser.add_argument("--keepon", default=False, action="store_true", help='keep on training')

    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")

    args = parser.parse_args()
    config = get_config()
    
    #args.gpu = 'cuda:' + args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #args.device = torch.device(args.gpu)
    args.device = torch.device('cuda:0')

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)    
    
    main(config, args)
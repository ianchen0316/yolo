""" YOLO-v3 Training Stage """

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.parse_config import parse_data_config
from utils.utils import load_classes
from models.darknet import Darknet


if __name__ == '__main__':
    
    # ================ Argument Parsing ===============================
    
    parser = argparse.ArgumentParser('YOLO v3')
    
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training')
    parser.add_argument('--gradient_accumulations', type=int, default=2, help='number of batches to accumulate gradient for each optimization step')
    parser.add_argument('--model_config', type=str, default='./config/yolov3.cfg', help='path to model config file')
    parser.add_argument('--data_config', type=str, default='./config/pascal_voc.data', help='path to data config file')
    parser.add_argument('--pretrained', type=str, help='if specified, starts from the checkpoint model')
    parser.add_argument('--img_size', type=int, default=416, help='resolution of each image')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval to save model weights')
    parser.add_argument('--compute_mAP', type=bool, default=False, help='if True, compute mAP every 10 batch')
    parser.add_argument('--multiscale_training', type=bool, default=True, help='allow for multi-scale training')
    
    args = parser.parse_args()
    
    # ================== Set device and directories =====================
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    
    # =================== Data Configuration ============================
    data_config = parse_data_config(args.data_config)
    train_path = data_config['train']
    valid_path = data_config['valid']
    class_names = load_classes(data_config['names'])
    
    # =================== Model Setup ===================================
    model = Darknet(args.model_config).to(device)
    
    
    
    
    
    
    
    
    
    


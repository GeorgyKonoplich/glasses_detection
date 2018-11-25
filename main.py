import argparse
from model import train
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model', help='model name')
    parser.add_argument('--dataset', type=str, default='dataset/celebA.zip', help='path to dataset')
    parser.add_argument('--model_folder', type=str, default='models', help='directory name to save model')
    parser.add_argument('--use_kfold', type=str2bool, default=False, help='using kfold')
    parser.add_argument('--epoch', type=int, default=2, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=32, help='The size of batch size')
    parser.add_argument('--img_size', type=int, default=128, help='The size of image')
    parser.add_argument('--augment_flag', type=str2bool, default=False, help='Image augmentation use or not')
    return parser.parse_args()


"""main"""
def main():
    args = parse_args()
    if args is None:
      exit()
    train(args)

if __name__ == '__main__':
    main()

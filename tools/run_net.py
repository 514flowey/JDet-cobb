import argparse
import os

from jdet.runner import Runner 
from jdet.config import init_cfg


def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="whole",
        help="train,val,test,whole",
        type=str,
    )
    args = parser.parse_args()

    assert args.task in ["train","val","test","whole"],f"{args.task} not support, please choose [train,val,test,whole]"
    
    if args.config_file:
        init_cfg(args.config_file)
    
    runner = Runner()

    if args.task == "whole":
        runner.run()
    elif args.task == "train":
        runner.train()
    elif args.task == "val":
        runner.val()
    elif args.task == "test":
        runner.test()

if __name__ == "__main__":
    main()
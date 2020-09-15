#-*- coding:utf-8 -*-
import os

if __name__ == "__main__":
        
    # command_pipe = ["python utility/processing.py --input=./data/train.csv --eda=True", \
    #                 "python utility/augment.py --input=./data/eda/train_1.txt --num_aug=8 --alpha=0.05", \
    #                 "python utility/augment.py --input=./data/eda/train_2.txt --num_aug=4 --alpha=0.05", \
    #                 "python utility/augment.py --input=./data/eda/train_3.txt --num_aug=4 --alpha=0.05", \
    #                 "python utility/augment.py --input=./data/eda/train_4.txt --num_aug=1 --alpha=0.01", \
    #                 "python utility/processing.py --input=./data/train.csv", \
    #                 "python utility/processing.py --input=./data/test.csv", \
    #                 "python utility/processing.py --input=./data/eda_train.csv "]
                    
    command_pipe = ["python utility/processing.py --input=./data/train.csv --eda=True", \
                    "python utility/augment.py --input=./data/eda/train_1.txt --num_aug=4 --alpha=0.05", \
                    "python utility/augment.py --input=./data/eda/train_2.txt --num_aug=2 --alpha=0.05", \
                    "python utility/augment.py --input=./data/eda/train_3.txt --num_aug=2 --alpha=0.05", \
                    "python utility/augment.py --input=./data/eda/train_4.txt --num_aug=1 --alpha=0.01", \
                    "python utility/processing.py --input=./data/eda_train.csv", \
                    "python main.py --mode=eda --ensemble=True"]
                    
    for command in command_pipe:
        os.system(command)

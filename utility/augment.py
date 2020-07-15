#-*- coding:utf-8 -*-
# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou
# https://github.com/jasonwei20/eda_nlp
import pandas as pd
from eda import *

#arguments to be parsed from command line
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha", required=False, type=float, help="percent of words in each sentence to be changed")
args = ap.parse_args()

#the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join(dirname(args.input), 'eda_' + basename(args.input))

#number of augmented sentences to generate per original sentence
num_aug = 9 #default
if args.num_aug:
    num_aug = args.num_aug

#how much to change each sentence
alpha = 0.1#default
if args.alpha:
    alpha = args.alpha

#generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha, num_aug=9):

    writer = open(output_file, 'w', encoding='UTF8')
    lines = open(train_orig, 'r', encoding='UTF8').readlines()

    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')
        label = parts[0]
        sentence = parts[1]
        id = parts[2]
        aug_sentences = eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        for aug_sentence in aug_sentences:
            writer.write(label + "\t" + aug_sentence + "\t" + id + '\n')

    writer.close()
    print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(num_aug))


if __name__ == "__main__":
    
    import os
    if not os.path.exists(args.input):
        os.makedirs(dirname(args.input))
        train = pd.read_csv('./data/train.csv')
        train.drop(5886, axis=0, inplace=True)
        for i in range(1, 5):
            title_df = train.groupby('median_relevance').get_group(i)[['product_title', 'id']]
            with open(join(dirname(args.input), "train_%d.txt"%i), 'w', encoding="utf-8") as f:
                for j in range(len(title_df)):
                    f.write("%d\t%s\t%d\n"%(i, title_df.iloc[j][0], title_df.iloc[j][1]))

    # generate augmented sentences and output into a new file
    # python utility/augment.py --input=./data/eda/train_1.txt --num_aug=16 --alpha=0.05
    # python utility/augment.py --input=./data/eda/train_2.txt --num_aug=8 --alpha=0.05
    # python utility/augment.py --input=./data/eda/train_3.txt --num_aug=8 --alpha=0.05
    # python utility/augment.py --input=./data/eda/train_4.txt --num_aug=2 --alpha=0.01
    gen_eda(args.input, output, alpha=alpha, num_aug=num_aug)

    train = pd.read_csv('./data/train.csv')
    df = pd.DataFrame(columns=['id', 'product_title', 'median_relevance'])
    with open(output, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line[:-1].split('\t')
            df.loc[i] = [parts[2], parts[1], parts[0]]
    
    train_i = train.groupby('median_relevance').get_group(int(args.input[-5]))
    df['query'] = train_i.loc[train_i.index.repeat(num_aug+1)].reset_index(drop=True)['query']
    df.to_csv('./data/eda_train_%d.csv'%int(args.input[-5]), index=False)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
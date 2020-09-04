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
def gen_eda(train_orig, output_file, alpha, num_aug=9, query_eda=False):

    writer = open(output_file, 'w', encoding='UTF8')
    lines = open(train_orig, 'r', encoding='UTF8').readlines()

    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')
        label = parts[0]
        id = parts[1]
        query = parts[2]
        title = parts[3]
        description = parts[4]
        if query_eda:
            aug_querys = eda(query, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        else :
            aug_querys = [query] * (num_aug+1)
        aug_titles = eda(title, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        aug_descriptions = eda(description, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        for aug_query, aug_title, aug_description in zip(aug_querys, aug_titles, aug_descriptions):
            writer.write(label + "\t" + id + "\t" + aug_query + "\t" + aug_title + "\t" + aug_description + '\n')

    writer.close()
    print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(num_aug))


if __name__ == "__main__":
    
    import os    
    if not os.path.exists(args.input):
        if not os.path.exists(os.path.dirname(args.input)):
            print('makedirs %s'%os.path.dirname(args.input))
            os.makedirs(os.path.dirname(args.input))
        train = pd.read_csv('./data/for_eda_train.csv')
        train.drop(5886, axis=0, inplace=True)
        for i in range(1, 5):
            title_df = train.groupby('median_relevance').get_group(i)[['id', 'query_preprocessed', 'product_title_preprocessed', 'product_description_preprocessed']]
            with open(os.path.join(os.path.dirname(args.input), "train_%d.txt"%i), 'w', encoding="utf-8") as f:
                for j in range(len(title_df)):
                    f.write("%d\t%d\t%s\t%s\t%s\n"%(i, title_df.iloc[j][0], title_df.iloc[j][1], title_df.iloc[j][2], title_df.iloc[j][3]))

    # generate augmented sentences and output into a new file
    gen_eda(args.input, output, alpha=alpha, num_aug=num_aug, query_eda=False)
    
    df = pd.DataFrame(columns=['median_relevance', 'id', 'query_preprocessed', 'product_title_preprocessed', 'product_description_preprocessed'])
    with open(output, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line[:-1].split('\t')
            df.loc[i] = [parts[0], parts[1], parts[2], parts[3],  parts[4]]
            
    # train = pd.read_csv('./data/train.csv')
    # train_i = train.groupby('median_relevance').get_group(int(args.input[-5]))
    # df['query'] = train_i.loc[train_i.index.repeat(num_aug+1)].reset_index(drop=True)['query']
    df.to_csv(os.path.join(os.path.dirname(output), 'eda_train_%d.csv'%int(args.input[-5])), index=False)
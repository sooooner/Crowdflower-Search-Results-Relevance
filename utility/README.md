# utility
사용자 함수 모듈

## Data preprocessing
전처리 
+ processing.py
  + remove_pattern(HTML-tag, URL, etc...)
  + remove_sw
  + tokenizer
  + lemmatizer

**Usage**
```
python utility/processing.py --input=./data/train.csv --eda=True
```

## Easy Data Augmentation

+ augment.py
+ eda.py

**Usage**
```
python utility/augment.py --input=./data/eda/train_1.txt --num_aug=4 --alpha=0.05 
python utility/augment.py --input=./data/eda/train_2.txt --num_aug=2 --alpha=0.05
python utility/augment.py --input=./data/eda/train_3.txt --num_aug=2 --alpha=0.05
python utility/augment.py --input=./data/eda/train_4.txt --num_aug=1 --alpha=0.01
```

**source**
[EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://github.com/jasonwei20/eda_nlp)

## utility
+ utility.py
  + metric(kappa, cos sim, etc...)
  + similarlity_stack
  + tf_weight_stack
  + plot_multiclass_roc_prc





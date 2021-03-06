# Predict the relevance of search results from E-Commerce

Kaggle [Crowdflower Search Results Relevance data](https://www.kaggle.com/c/crowdflower-search-relevance)를 이용한 E-Commerce 사용자 검색 시스템 만족도 예측 모델

## Description
프로젝트 과정 설명(링크는 **Medium**글)

+ Description.ipynb
  + TF-IDF, LSA, SVM, Word2vec를 사용한 E-Commerce 사용자 만족도 예측 모델 
  + [검색 서비스 만족도 판별모델(1)](https://medium.com/@tnsgh0101/crowdflower-search-results-relevance-with-lsa-2b05ac8b1a23)
  + [검색 서비스 만족도 판별모델(2)](https://medium.com/@tnsgh0101/%EA%B2%80%EC%83%89-%EC%84%9C%EB%B9%84%EC%8A%A4-%EB%A7%8C%EC%A1%B1%EB%8F%84-%ED%8C%90%EB%B3%84%EB%AA%A8%EB%8D%B8-2-9a8411baecab)

## Project structure
This project is organized as follows.

```
.
└── utility/           
    ├── README.md 
    ├── __init__.py
    ├── augment.py                     # Data augmentation function
    ├── eda.py                         # EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks
    ├── processing.py                  # Preprocessing functions
    ├── predict.py                     # predict functions
    └── utility.py                     # metrics, distance stack, plot, etc. functions
└── example/      
    ├── Description.ipynb              # description for this project
    ├── EDA.ipynb                      # Exploratory Data Analysis
    ├── LSA.ipynb                      # The whole process of the project
    ├── preprocessing.ipynb            # Data preprocessing process flow
    └── word2vec.ipynb                 # Implementing and applying word2vec(*Implemented in tensorflow 1 version.)
├── .gitignore       
├── README.md
├── gridsearch.py                      # Parallelized gridsearchCV to find hyperparameters
└── main.py                            # Make submission py

```
  
## Exploratory Data Analysis
데이터 탐색
+ EDA.ipynb

## Preprocessing
데이터 전처리 과정
+ preprocessing.ipynb
+ utility/processing.py

## Data Augmentation
데이터 증강
+ utility/augment.py
+ utility/eda.py

## model implement
사용자 만족도 판별 모델링 과정
+ LSA.ipynb
+ word2vec.ipynb

## utility
모델에 사용된 함수  
utility/[README.md](https://github.com/sooooner/Crowdflower-Search-Results-Relevance/blob/master/utility/README.md) 참고

## Usage
processing.py 로 데이터 전처리후 data Augmentation을 합니다.(data Augmentation의 각 하이퍼파라미터는 논문을 따릅니다) Augmentation main.py로 submission을 생성합니다.

```
python utility/processing.py --input=./data/train.csv --eda=True
python utility/augment.py --input=./data/eda/train_1.txt --num_aug=8 --alpha=0.05
python utility/augment.py --input=./data/eda/train_2.txt --num_aug=4 --alpha=0.05
python utility/augment.py --input=./data/eda/train_3.txt --num_aug=4 --alpha=0.05
python utility/augment.py --input=./data/eda/train_4.txt --num_aug=0
python utility/processing.py --input=./data/eda_train.csv
python main.py --mode=eda 
```

## 참고자료 

1. [Predicting the Relevance of Search Results for E-Commerce Systems](https://www.researchgate.net/publication/286219675_Predicting_the_Relevance_of_Search_Results_for_E-Commerce_Systems)  
2. [Using TF-IDF to determine word relevance in document queries](https://www.researchgate.net/publication/228818851_Using_TF-IDF_to_determine_word_relevance_in_document_queries)  
3. [Classifying Positive or Negative Text Using Features Based on Opinion Words and Term Frequency - Inverse Document Frequenc](https://ieeexplore.ieee.org/document/8541274)  
4. [An introduction to latent semantic analysis](https://www.tandfonline.com/doi/abs/10.1080/01638539809545028)  
5. [Using Linear Algebra for Intelligent Information Retrieval](https://epubs.siam.org/doi/abs/10.1137/1037127?journalCode=siread)  
6. [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/abs/1901.11196)  
7. [Weighted kappa loss function for multi-class classification of ordinal data in deep learning](https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666)  
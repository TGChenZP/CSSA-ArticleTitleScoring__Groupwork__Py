# CSSA_ArticleTitleScoring__Groupwork__Py

This project was undertaken by our student club's Data Science team and aims to analyse what contribute to good article publicity on our club's social media account, using advanced NLP techniques.

### Team members:
- [Lang (Ron) Chen](https://github.com/TGChenZP): Project Lead and Experiment Design 
- Yuchen (Steven) Luo: Data cleaning and visualisation
- [Angqi (Devin) Meng](https://github.com/DDDDDDDEVIN): Data cleaning and Experiments
- [Ruonan (Selena) Xiong](https://github.com/ruonannn): Experiments
- [Mingqi He](https://github.com/mq-he): Experiment Design
- [Mary Huang](https://github.com/MaryHuang8): Experiments
- [Melanie Huang](https://github.com/MelanieH7): Experiments
- [Shan Lin](https://github.com/skylar9503): Experiment Design
- [Zhuoya Zhou](https://github.com/ykforever0504): Experiment Design

## Themes:
NLP, Transfer Learning, BERT, Machine Learning

## Blog Post:
https://mp.weixin.qq.com/s/XXhjVXS4fbvHBjBLwWRj-w

## Dataset
- 236 articles from CSSA's public account (March 2021–March 2024), each with Title text, WeChat PengYouQuan (PYQ) text, reposts, and reads (in Chinese).

## Method
Preprocessing
1.	Removed emojis and unrecognizable characters from text data.
Feature Engineering & Train-Test Split
2. Reposts encoded as a Boolean (>0).
3. Used GanymedeNil's text2vec-large-chinese (pre-trained BERT) to extract 768-dimensional embeddings for Title and PYQ text.
4. Split data into training (70%), validation (15%), and test (15%) sets.
5. Applied PCA on the training set to reduce dimensions from 768 to 32 for both Title and PYQ text; transformation applied to validation and test sets.
6. Created 'Read Score' by min-max normalising the rank of articles based on reads.
Modelling
7. Modelled rank prediction (regression) in two experiments: (1) Using 32 Title text features and reposts, and (2) Using 32 Title + 32 PYQ text features and reposts.
    - models: ADABoost, CatBoost, ExplainableBoost, ExtraRandomForest, Gaussian Process Regressor, GradientBoost, HGBoost, Lasso Regression, LightGradientBoost, Linear Regression, NU SVM (Polynomial Kernel), NU SVM (RBF Kernel), Random Forest, Ridge Regression, SVM (Polynomial Kernel), SVM (RBF Kernel), and XGBoost.
Evaluation
8. Selected the best model from experiment 1 based on the highest validation R², reporting its test performance in both experiments.
9. Conducted feature importance and SHAP analysis to interpret the model.
10. Investigated article themes with extreme feature values via SHAP.

## Results
| Model                        | Embedding   | Features   |   Val R2 |   Test R2 |
|:-----------------------------|:------------|:-----------|---------:|----------:|
| ADABoost                     | Text2Vec    | Title      |    0.118 |   -0.014  |
| CatBoost                     | Text2Vec    | Title      |    **0.192** |   -0.178  |
| ExplainableBoost             | Text2Vec    | Title      |    0.148 |    0.1    |
| ExtraRandomForest            | Text2Vec    | Title      |    0.148 |   -0.091  |
| Gaussian Processor Regressor | Text2Vec    | Title      |   -2.796 |   -3.858  |
| GradientBoost                | Text2Vec    | Title      |    0.162 |   -0.022  |
| HGBooost                     | Text2Vec    | Title      |    0.106 |   -0.091  |
| Lasso Regression             | Text2Vec    | Title      |    0.071 |    0.068  |
| LightGradientBoost           | Text2Vec    | Title      |    0.103 |    0.034  |
| Linear Regression            | Text2Vec    | Title      |    0.069 |    0.087  |
| NU SVM (Poly)                | Text2Vec    | Title      |    0.126 |    0.019  |
| NU SVM (RBF)                 | Text2Vec    | Title      |    0.126 |    0.019  |
| Random Forest                | Text2Vec    | Title      |    0.155 |    0.085  |
| Ridge Regression             | Text2Vec    | Title      |    0.06  |   -0.017  |
| SVM (Poly)                   | Text2Vec    | Title      |    0.114 |    0.031  |
| SVM (RBF)                    | Text2Vec    | Title      |    0.114 |    0.031  |
| XGBoost                      | Text2Vec    | Title      |    0.118 |    0.078  |
| ADABoost                     | Text2Vec    | Title&PYQ  |    0.224 |    0.042  |
| CatBoost                     | Text2Vec    | Title&PYQ  |    **0.261** |    **0.164**  |
| ExplainableBoost             | Text2Vec    | Title&PYQ  |    0.16  |   -0.167  |
| ExtraRandomForest            | Text2Vec    | Title&PYQ  |    0.146 |   -0.007  |
| Gaussian Processor Regressor | Text2Vec    | Title&PYQ  |   -3.12  |   -4.203  |
| GradientBoost                | Text2Vec    | Title&PYQ  |    0.342 |    0.07   |
| HGBooost                     | Text2Vec    | Title&PYQ  |    0.153 |   -0.063  |
| Lasso Regression             | Text2Vec    | Title&PYQ  |    0.251 |    0.106  |
| LightGradientBoost           | Text2Vec    | Title&PYQ  |    0.209 |    0.113  |
| Linear Regression            | Text2Vec    | Title&PYQ  |    0.251 |    0.106  |
| NU SVM (Poly)                | Text2Vec    | Title&PYQ  |    0.299 |   -0.083  |
| NU SVM (RBF)                 | Text2Vec    | Title&PYQ  |    0.299 |   -0.083  |
| Random Forest                | Text2Vec    | Title&PYQ  |    0.101 |    0.052  |
| Ridge Regression             | Text2Vec    | Title&PYQ  |    0.251 |    0.106  |
| SVM (Poly)                   | Text2Vec    | Title&PYQ  |    0.254 |    0.052  |
| SVM (RBF)                    | Text2Vec    | Title&PYQ  |    0.254 |    0.052  |
| XGBoost                      | Text2Vec    | Title&PYQ  |    0.189 |    0.114  |
# CSSA_ArticleTitleScoring__Groupwork__Py

This project was undertaken by our student club's Data Science team and aims to analyse what contribute to good article publicity on our club's social media account, using advanced NLP techniques.

### Team members:
- Lang (Ron) Chen: Project Lead and Experiment Design
- Yuchen (Steven) Luo: Data cleaning and visualisation
- Angqi (Devin) Meng: Data cleaning and Experiments
- Ruonan (Selena) Xiong: Experiments
- Mingqi Hao: Experiment Design
- Mary Huang: Experiments
- Melanie Huang: Experiments
- Shan Lin: Experiment Design

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


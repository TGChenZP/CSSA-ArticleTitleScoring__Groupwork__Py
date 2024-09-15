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
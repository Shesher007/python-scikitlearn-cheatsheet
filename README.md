---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# scikit-learn - Machine Learning in Python

* Simple and efficient tools for predictive data analysis
* Accessible to everybody, and reusable in various contexts
* Built on NumPy, SciPy, and matplotlib
* Open source, commercially usable - BSD license

> Regressions ++ Classifications ++ Clustering ++ Dimensionality Reduction ++ Model Selection ++ Pre-processing

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.datasets import load_iris, load_wine, fetch_20newsgroups, fetch_openml
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    ElasticNet
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    cross_validate
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OrdinalEncoder,
    LabelEncoder,
    OneHotEncoder,
    PolynomialFeatures
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
```

## Working with Missing Values

```python
X_missing = pd.DataFrame(
    np.array([5,2,3,np.NaN,np.NaN,4,-3,2,1,8,np.NaN,4,10,np.NaN,5]).reshape(5,3)
)
X_missing.columns = ['f1','f2','f3']

X_missing
```

|  | f1 | f2 | f3 |
| -- | -- | -- | -- |
| 0 | 5.0 | 2.0 | 3.0 |
| 1 | NaN | NaN | 4.0 |
| 2 | -3.0 | 2.0 | 1.0 |
| 3 | 8.0 | NaN | 4.0 |
| 4 | 10.0 | NaN | 5.0 |

```python
X_missing.isnull().sum()

# f1    1
# f2    3
# f3    0
# dtype: int64
```

### Missing Indicator

```python
indicator = MissingIndicator(missing_values=np.NaN)
indicator = indicator.fit_transform(X_missing)
indicator = pd.DataFrame(indicator, columns=['a1', 'a2'])
indicator
```

|  | a1 | a2 |
| -- | -- | -- |
| 0 | False | False |
| 1 | True | True |
| 2 | False | False |
| 3 | False | True |
| 4 | False | True |


### Simple Imputer

```python
imputer_mean = SimpleImputer(missing_values=np.NaN, strategy='mean')
X_filled_mean = pd.DataFrame(imputer_mean.fit_transform(X_missing))
X_filled_mean.columns = ['f1','f2','f3']
X_filled_mean
```

|  | f1 | f2 | f3 |
| -- | -- | -- | -- |
| 0 | 5.0 | 2.0 | 3.0 |
| 1 | 5.0 | 2.0 | 4.0 |
| 2 | -3.0 | 2.0 | 1.0 |
| 3 | 8.0 | 2.0 | 4.0 |
| 4 | 10.0 | 2.0 | 5.0 |

```python
imputer_median = SimpleImputer(missing_values=np.NaN, strategy='median')
X_filled_median = pd.DataFrame(imputer_median.fit_transform(X_missing))
X_filled_median.columns = ['f1','f2','f3']
X_filled_median
```

|  | f1 | f2 | f3 |
| -- | -- | -- | -- |
| 0 | 5.0 | 2.0 | 3.0 |
| 1 | 6.5 | 2.0 | 4.0 |
| 2 | -3.0 | 2.0 | 1.0 |
| 3 | 8.0 | 2.0 | 4.0 |
| 4 | 10.0 | 2.0 | 5.0 |

```python
imputer_median = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
X_filled_median = pd.DataFrame(imputer_median.fit_transform(X_missing))
X_filled_median.columns = ['f1','f2','f3']
X_filled_median
```

|  | f1 | f2 | f3 |
| -- | -- | -- | -- |
| 0 | 5.0 | 2.0 | 3.0 |
| 1 | -3.0 | 2.0 | 4.0 |
| 2 | -3.0 | 2.0 | 1.0 |
| 3 | 8.0 | 2.0 | 4.0 |
| 4 | 10.0 | 2.0 | 5.0 |


### Drop Missing Data

```python
X_missing_dropped = X_missing.dropna(axis=1)
X_missing_dropped
```

|  | f3 |
| -- | -- |
| 0 | 3.0 |
| 1 | 4.0 |
| 2 | 1.0 |
| 3 | 4.0 |
| 4 | 5.0 |

```python
X_missing_dropped = X_missing.dropna(axis=0).reset_index()
X_missing_dropped
```

|   | f1 | f2 | f3 |
| -- | -- | -- | -- |
| 0 | 5.0 | 2.0 | 3.0 |
| 1 | -3.0 | 2.0 | 1.0 |


## Categorical Data Preprocessing

```python
X_cat_df = pd.DataFrame(
    np.array([
        ['M', 'O-', 'medium'],
        ['M', 'O-', 'high'],
        ['F', 'O+', 'high'],
        ['F', 'AB', 'low'],
        ['F', 'B+', 'medium']
    ])
)

X_cat_df.columns = ['f1','f2','f3']

X_cat_df
```

|  | f1 | f2 | f3 |
| -- | -- | -- | -- |
| 0 | M | O- | medium |
| 1 | M | O- | high |
| 2 | F | O+ | high |
| 3 | F | AB | low |
| 4 | F | B+ | medium |


### Ordinal Encoder

```python
encoder_ord = OrdinalEncoder(dtype='int')

X_cat_df.f3 = encoder_ord.fit_transform(X_cat_df.f3.values.reshape(-1, 1))
X_cat_df
```

|  | f1 | f2 | f3 |
| -- | -- | -- | -- |
| 0 | M | O- | 2 |
| 1 | M | O- | 0 |
| 2 | F | O+ | 0 |
| 3 | F | AB | 1 |
| 4 | F | B+ | 2 |


### Label Encoder

```python
encoder_lab = LabelEncoder()
X_cat_df['f2'] = encoder_lab.fit_transform(X_cat_df['f2'])
X_cat_df
```

|  | f1 | f2 | f3 |
| -- | -- | -- | -- |
| 0 | M | 3 | 2 |
| 1 | M | 3 | 0 |
| 2 | F | 2 | 0 |
| 3 | F | 0 | 1 |
| 4 | F | 1 | 2 |


### OneHot  Encoder

```python
encoder_oh = OneHotEncoder(dtype='int')

onehot_df = pd.DataFrame(
    encoder_oh.fit_transform(X_cat_df[['f1']])
    .toarray(),
    columns=['F', 'M']
)

onehot_df['f2'] = X_cat_df.f2
onehot_df['f3'] = X_cat_df.f3
onehot_df
```

|   | F | M | f2 | f3 |
| -- | -- | -- | -- | -- |
| 0 | 0 | 1 | 3 | 2 |
| 1 | 0 | 1 | 3 | 0 |
| 2 | 1 | 0 | 2 | 0 |
| 3 | 1 | 0 | 0 | 1 |
| 4 | 1 | 0 | 1 | 2 |


## Loading SK Datasets

### Toy Datasets

|  |  |  |
| -- | -- | -- |
| load_iris(*[, return_X_y, as_frame]) | classification | Load and return the iris dataset. |
| load_diabetes(*[, return_X_y, as_frame, scaled]) | regression | Load and return the diabetes dataset. |
| load_digits(*[, n_class, return_X_y, as_frame]) | classification |  Load and return the digits dataset. |
| load_linnerud(*[, return_X_y, as_frame]) | multi-output regression | Load and return the physical exercise Linnerud dataset. |
| load_wine(*[, return_X_y, as_frame]) | classification | Load and return the wine dataset. | 
| load_breast_cancer(*[, return_X_y, as_frame]) | classification | Load and return the breast cancer wisconsin dataset. |

```python
iris_ds = load_iris()
iris_data = iris_ds.data
col_names = iris_ds.feature_names
target_names = iris_ds.target_names

print(
    'Iris Dataset',
    '\n * Data array: ',
    iris_data.shape,
    '\n * Column names: ',
    col_names,
    '\n * Target names: ',
    target_names
)

# Iris Dataset 
#  * Data array:  (150, 4) 
#  * Column names:  ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] 
#  * Target names:  ['setosa' 'versicolor' 'virginica']
```

```python
iris_df = pd.DataFrame(data=iris_data, columns=col_names)

iris_df.head()
```

|  | sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) |
| -- | -- | -- | -- | -- |
| 0 | 5.1 | 3.5 | 1.4 | 0.2 |
| 1 | 4.9 | 3.0 | 1.4 | 0.2 |
| 2 | 4.7 | 3.2 | 1.3 | 0.2 |
| 3 | 4.6 | 3.1 | 1.5 | 0.2 |
| 4 | 5.0 | 3.6 | 1.4 | 0.2 |


### Real World Datasets

|  |  |  |
| -- | -- | -- |
| fetch_olivetti_faces(*[, data_home, ...]) | classification | Load the Olivetti faces data-set from AT&T. |
| fetch_20newsgroups(*[, data_home, subset, ...]) | classification | Load the filenames and data from the 20 newsgroups dataset. |
| fetch_20newsgroups_vectorized(*[, subset, ...]) | classification | Load and vectorize the 20 newsgroups dataset. |
| fetch_lfw_people(*[, data_home, funneled, ...]) | classification | Load the Labeled Faces in the Wild (LFW) people dataset. |
| fetch_lfw_pairs(*[, subset, data_home, ...]) | classification | Load the Labeled Faces in the Wild (LFW) pairs dataset. |
| fetch_covtype(*[, data_home, ...]) | classification | Load the covertype dataset. |
| fetch_rcv1(*[, data_home, subset, ...]) | classification | Load the RCV1 multilabel dataset. |
| fetch_kddcup99(*[, subset, data_home, ...]) | classification | Load the kddcup99 dataset. |
| fetch_california_housing(*[, data_home, ...]) | regression | Load the California housing dataset. |

```python
newsgroups_train = fetch_20newsgroups(subset='train')
train_data = newsgroups_train.data
col_names = newsgroups_train.filenames.shape
target_names = newsgroups_train.target.shape

print(
    'Newsgroup - Train Subset',
    '\n * Data array: ',
    len(train_data),
    '\n * Column names: ',
    col_names,
    '\n * Target names: ',
    target_names
)

# Newsgroup - Train Subset 
#  * Data array:  11314 
#  * Column names:  (11314,) 
#  * Target names:  (11314,)
```

```python
print('Target Names: ', newsgroups_train.target_names)

# Target Names:  ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
```

### OpenML Datasets

* [openml.org](https://openml.org/search?type=data&sort=runs&status=active)
* [Mice Protein Dataset](https://openml.org/search?type=data&status=active&id=40966)

```python
mice_ds = fetch_openml(name='miceprotein', version=4, parser="auto")
```

```python
print(
    'Mice Protein Dataset',
    '\n * Data Shape: ',
    mice_ds.data.shape,
    '\n * Target Shape: ',
    mice_ds.target.shape,
    '\n * Target Names: ',
    np.unique(mice_ds.target)
)

# Mice Protein Dataset 
#  * Data Shape:  (1080, 77) 
#  * Target Shape:  (1080,) 
#  * Target Names:  ['c-CS-m' 'c-CS-s' 'c-SC-m' 'c-SC-s' 't-CS-m' 't-CS-s' 't-SC-m' 't-SC-s']

```

```python
print(mice_ds.DESCR)
```

## Regression Models

### Simple Linear Regression

```python
iris_df.plot(
    figsize=(12,5),
    kind='scatter',
    x='sepal length (cm)',
    y='sepal width (cm)',
    title='Iris Dataset :: Sepal Width&Height'
)

print(iris_df.corr())
```

> The __Sepal Width__ has very little correlation to all other metrics but itself. While the other three correlate nicely:

|  | sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) |
| -- | -- | -- | -- | -- |
| sepal length (cm) | 1.000000 | -0.117570 | 0.871754 | 0.817941 |
| sepal width (cm) | -0.117570 | 1.000000 | -0.428440 | -0.366126 |
| petal length (cm) | 0.871754 | -0.428440 | 1.000000 | 0.962865 |
| petal width (cm) | 0.817941 | -0.366126 | 0.962865 | 1.000000 |

![scikit-learn - Machine Learning in Python](./assets/Scikit_Learn_01.webp)

![scikit-learn - Machine Learning in Python](./assets/Scikit_Learn_02.webp)


#### Data Pre-processing

```python
iris_df['petal length (cm)'][:1]
# 0    1.4
# Name: petal length (cm), dtype: float64
```

```python
iris_df['petal length (cm)'].values.reshape(-1,1)[:1]
# array([[1.4]])
```

```python
# scikit expects a 2s imput => remove index
X = iris_df['petal length (cm)'].values.reshape(-1,1)
y = iris_df['petal width (cm)'].values.reshape(-1,1)
```

```python
# train/test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape, X_test.shape)
# (120, 1) (30, 1) 80:20 split
```

#### Model Training

```python
regressor = LinearRegression()
regressor.fit(X_train,y_train)

intercept = regressor.intercept_
slope = regressor.coef_

print(' Intercept: ', intercept, '\n Slope: ', slope)
#  Intercept:  [-0.35135666] 
#  Correlation Coeficient:  [[0.41310505]]
```

#### Predictions

```python
y_pred = regressor.predict([X_test[0]])
print(' Prediction: ', y_pred, '\n True Value: ', y_test[0])
#  Prediction:  [[0.22699041]] 
#  True Value:  [0.2]
```

```python
def predict(value):
    return (slope*value + intercept)[0][0]
```

```python
print('Prediction: ', predict(X_test[0]))
# Prediction:  [[0.22699041]]
```

```python
iris_df['petal width (cm) prediction'] = iris_df['petal length (cm)'].apply(predict)
print(' Prediction: ', iris_df['petal width (cm) prediction'][0], '\n True Value: ', iris_df['petal width (cm)'][0])
#  Prediction:  0.22699041280334376 
#  True Value:  0.2
```

```python
iris_df.head(10)
```

|   | sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | petal width (cm) prediction |
| -- | -- | -- | -- | -- | -- |
| 0 | 5.1 | 3.5 | 1.4 | 0.2 | 0.226990 |
| 1 | 4.9 | 3.0 | 1.4 | 0.2 | 0.226990 |
| 2 | 4.7 | 3.2 | 1.3 | 0.2 | 0.185680 |
| 3 | 4.6 | 3.1 | 1.5 | 0.2 | 0.268301 |
| 4 | 5.0 | 3.6 | 1.4 | 0.2 | 0.226990 |
| 5 | 5.4 | 3.9 | 1.7 | 0.4 | 0.350922 |
| 6 | 4.6 | 3.4 | 1.4 | 0.3 | 0.226990 |
| 7 | 5.0 | 3.4 | 1.5 | 0.2 | 0.268301 |
| 8 | 4.4 | 2.9 | 1.4 | 0.2 | 0.226990 |
| 9 | 4.9 | 3.1 | 1.5 | 0.1 | 0.268301 |

```python
iris_df.plot(
    figsize=(12,5),
    kind='scatter',
    x='petal width (cm)',
    y='petal width (cm) prediction',
    # no value in colorizing..just looks pretty
    c='petal width (cm) prediction',
    colormap='summer',
    title='Iris Dataset - Sepal Width True vs Prediction'
)
```

![scikit-learn - Machine Learning in Python](./assets/Scikit_Learn_03.webp)


#### Model Evaluation

```python
mae = mean_absolute_error(
    iris_df['petal width (cm)'],
    iris_df['petal width (cm) prediction']
)

mse = mean_squared_error(
    iris_df['petal width (cm)'],
    iris_df['petal width (cm) prediction']
)

rmse = np.sqrt(mse)

print(' MAE: ', mae, '\n MSE: ', mse, '\n RMSE: ', rmse)

#  MAE:  0.1569441318761155 
#  MSE:  0.04209214667485277 
#  RMSE:  0.2051637070118708
```

### ElasticNet Regression
#### Dataset

```python
!wget https://raw.githubusercontent.com/Satish-Vennapu/DataScience/main/AMES_Final_DF.csv -P datasets
```

```python
ames_df = pd.read_csv('datasets/AMES_Final_DF.csv')
ames_df.head(5).transpose()
```

|  | 0 | 1 | 2 | 3 | 4 |
| -- | -- | -- | -- | -- | -- |
| Lot Frontage | 141.0 | 80.0 | 81.0 | 93.0 | 74.0 |
| Lot Area | 31770.0 | 11622.0 | 14267.0 | 11160.0 | 13830.0 |
| Overall Qual | 6.0 | 5.0 | 6.0 | 7.0 | 5.0 |
| Overall Cond | 5.0 | 6.0 | 6.0 | 5.0 | 5.0 |
| Year Built | 1960.0 | 1961.0 | 1958.0 | 1968.0 | 1997.0 |
| ... |
| Sale Condition_AdjLand | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Sale Condition_Alloca | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Sale Condition_Family | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Sale Condition_Normal | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Sale Condition_Partial | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
_274 rows × 5 columns_

```python
# the target value is:
ames_df['SalePrice']
```

|  |  |
| -- | -- |
|0 | 215000 |
|1 | 105000 |
|2 | 172000 |
|3 | 244000 |
|4 | 189900 |
|    ...    |
|2920 | 142500 |
|2921 | 131000 |
|2922 | 132000 |
|2923 | 170000 |
|2924 | 188000 |
_Name: SalePrice, Length: 2925, dtype: int64_


#### Preprocessing

```python
# remove target column from training dataset
X_ames = ames_df.drop('SalePrice', axis=1)
y_ames = ames_df['SalePrice']

print(X_ames.shape, y_ames.shape)
# (2925, 273) (2925,)
```

```python
# train/test split
X_ames_train, X_ames_test, y_ames_train, y_ames_test = train_test_split(
    X_ames,
    y_ames,
    test_size=0.1,
    random_state=101
)

print(X_ames_train.shape, X_ames_test.shape)
# (2632, 273) (293, 273)
```

```python
# normalize feature set
scaler = StandardScaler()
X_ames_train_scaled = scaler.fit_transform(X_ames_train)

X_ames_test_scaled = scaler.transform(X_ames_test)
```

#### Grid Search for Hyperparameters

```python
base_ames_elastic_net_model = ElasticNet(max_iter=int(1e4))
```

```python
param_grid = {
    'alpha': [50, 75, 100, 125, 150],
    'l1_ratio':[0.2, 0.4, 0.6, 0.8, 1.0]
}
```

```python
grid_ames_model = GridSearchCV(
    estimator=base_ames_elastic_net_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5, verbose=1
)

grid_ames_model.fit(X_ames_train_scaled, y_ames_train)

print(
    'Results:\nBest Estimator: ',
    grid_ames_model.best_estimator_,
    '\nBest Hyperparameter: ',
    grid_ames_model.best_params_
)
```

__Results__:
* Best Estimator:  `ElasticNet(alpha=125, l1_ratio=1.0, max_iter=10000)`
* Best Hyperparameter:  `{'alpha': 125, 'l1_ratio': 1.0}`


#### Model Evaluation

```python
y_ames_pred = grid_ames_model.predict(X_ames_test_scaled)

print(
    'MAE: ',
    mean_absolute_error(y_ames_test, y_ames_pred),
    'MSE: ',
    mean_squared_error(y_ames_test, y_ames_pred),
    'RMSE: ',
    np.sqrt(mean_squared_error(y_ames_test, y_ames_pred))
)

# MAE:  14185.506207185055 MSE:  422714457.5190704 RMSE:  20560.020854052418
```

```python
# average SalePrize
np.mean(ames_df['SalePrice'])
# 180815.53743589742

rel_error_avg = mean_absolute_error(y_ames_test, y_ames_pred) * 100 / np.mean(ames_df['SalePrice'])
print('Pridictions are on average off by: ', rel_error_avg.round(2), '%')
# Pridictions are on average off by:  7.85 %
```

```python
plt.figure(figsize=(10,4))

plt.scatter(y_ames_test,y_ames_pred, c='mediumspringgreen', s=3)
plt.axline((0, 0), slope=1, color='dodgerblue', linestyle=(':'))

plt.title('Prediction Accuracy :: MAE:'+ str(mean_absolute_error(y_ames_test, y_ames_pred).round(2)) + 'US$')
plt.xlabel('True Sales Price')
plt.ylabel('Predicted Sales Price')
plt.savefig('assets/Scikit_Learn_11.webp', bbox_inches='tight')
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_11.webp)


### Multiple Linear Regression

Above I used the `petal width` and `length` to create a linear regression model. But as explored earlier we can also use the `sepal length` (only the `sepal width` does not show a linear correlation):

```python
print(iris_df.corr())
```

|  | sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) |
| -- | -- | -- | -- | -- |
| sepal length (cm) | 1.000000 | -0.117570 | 0.871754 | 0.817941 |
| sepal width (cm) | -0.117570 | 1.000000 | -0.428440 | -0.366126 |
| petal length (cm) | 0.871754 | -0.428440 | 1.000000 | 0.962865 |
| petal width (cm) | 0.817941 | -0.366126 | 0.962865 | 1.000000 |

```python
X_multi = iris_df[['petal length (cm)', 'sepal length (cm)']]
y = iris_df['petal width (cm)']
```

```python
regressor_multi = LinearRegression()
regressor_multi.fit(X_multi, y)

intercept_multi = regressor_multi.intercept_
slope_multi = regressor_multi.coef_

print(' Intercept: ', intercept_multi, '\n Slope: ', slope_multi)

#  Intercept:  -0.00899597269816943 
#  Slope:  [ 0.44937611 -0.08221782]
```

```python
def predict_multi(petal_length, sepal_length):
    return (slope_multi[0]*petal_length + slope_multi[1]*sepal_length + intercept_multi)
```

```python
y_pred = predict_multi(
    iris_df['petal length (cm)'][0],
    iris_df['sepal length (cm)'][0]
)

print(' Prediction: ', y_pred, '\n True value: ', iris_df['petal width (cm)'][0])
#  Prediction:  0.20081970121763193 
#  True value:  0.2
```

```python
iris_df['petal width (cm) prediction (multi)'] = (
    (
        slope_multi[0] * iris_df['petal length (cm)']
    ) + (
        slope_multi[1] * iris_df['sepal length (cm)']
    ) + (
        intercept_multi
    ) 
)
```

```python
iris_df.head(10)
```

|    | sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | petal width (cm) prediction | petal width (cm) prediction (multi) |
| -- | -- | -- | -- | -- | -- | -- |
| 0 | 5.1 | 3.5 | 1.4 | 0.2 | 0.226990 | 0.200820 |
| 1 | 4.9 | 3.0 | 1.4 | 0.2 | 0.226990 | 0.217263 |
| 2 | 4.7 | 3.2 | 1.3 | 0.2 | 0.185680 | 0.188769 |
| 3 | 4.6 | 3.1 | 1.5 | 0.2 | 0.268301 | 0.286866 |
| 4 | 5.0 | 3.6 | 1.4 | 0.2 | 0.226990 | 0.209041 |
| 5 | 5.4 | 3.9 | 1.7 | 0.4 | 0.350922 | 0.310967 |
| 6 | 4.6 | 3.4 | 1.4 | 0.3 | 0.226990 | 0.241929 |
| 7 | 5.0 | 3.4 | 1.5 | 0.2 | 0.268301 | 0.253979 |
| 8 | 4.4 | 2.9 | 1.4 | 0.2 | 0.226990 | 0.258372 |
| 9 | 4.9 | 3.1 | 1.5 | 0.1 | 0.268301 | 0.262201 |

```python
iris_df.plot(
    figsize=(12,5),
    kind='scatter',
    x='petal width (cm)',
    y='petal width (cm) prediction (multi)',
    c='petal width (cm) prediction',
    colormap='summer',
    title='Iris Dataset - Sepal Width True vs Prediction (multi)'
)
```

![scikit-learn - Machine Learning in Python](./assets/Scikit_Learn_04.webp)

```python
mae_multi = mean_absolute_error(
    iris_df['petal width (cm)'],
    iris_df['petal width (cm) prediction (multi)']
)

mse_multi = mean_squared_error(
    iris_df['petal width (cm)'],
    iris_df['petal width (cm) prediction (multi)']
)

rmse_multi = np.sqrt(mse_multi)

print(' MAE_Multi: ', mae_multi,' MAE: ', mae, '\n MSE_Multi: ', mse_multi, ' MSE: ', mse, '\n RMSE_Multi: ', rmse_multi, ' RMSE: ', rmse)
```

The accuracy of the model was improved by adding an additional, correlating value:

|          | Multi Regression   | Single Regression |
| --       | --                 | --                |
| Mean Absolute Error | 0.15562108079300102 | 0.1569441318761155 |
| Mean Squared Error | 0.04096208526408982 | 0.04209214667485277 |
| Root Mean Squared Error | 0.20239092189149646 | 0.2051637070118708 |


## Logistic Regression Model

### Binary Logistic Regression

#### Dataset

```python
np.random.seed(666)

# generate 10 index values between 0-10
x_data_logistic_binary = np.random.randint(10, size=(10)).reshape(-1, 1)
# generate binary category for values above
y_data_logistic_binary = np.random.randint(2, size=10)
```

#### Model Fitting

```python
logistic_binary_model = LogisticRegression(
    solver='liblinear',
    C=10.0,
    random_state=0
)

logistic_binary_model.fit(x_data_logistic_binary, y_data_logistic_binary)

intercept_logistic_binary = logistic_binary_model.intercept_
slope_logistic_binary = logistic_binary_model.coef_

print(' Intercept: ', intercept_logistic_binary, '\n Slope: ', slope_logistic_binary)

#  Intercept:  [-0.4832956] 
#  Slope:  [[0.11180522]]
```

#### Model Predictions

```python
prob_pred_logistic_binary = logistic_binary_model.predict_proba(x_data_logistic_binary)
y_pred_logistic_binary = logistic_binary_model.predict(x_data_logistic_binary)


print('Prediction Probabilities: ', prob_pred[:1])

unique, counts = np.unique(y_pred_logistic_binary, return_counts=True)
print('Classes: ', unique, '| Number of Class Instances: ', counts)

# probabilities e.g. below -> 58% certainty that the first element is class 0

# Prediction Probabilities:  [[0.58097284 0.41902716]]
# Classes:  [0 1] | Number of Class Instances:  [5 5]
```

#### Model Evaluation

```python
conf_mtx = confusion_matrix(y_data_logistic_binary, y_pred_logistic_binary)
conf_mtx

# [2, 3] [TP, FP]
# [3, 2] [FN, TN]
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/confusion-matrix.webp)

```python
report = classification_report(y_data_logistic_binary, y_pred_logistic_binary)
print(report)
```

|              | precision | recall | f1-score | support |
|    --        |  --  |  --  |  --  |  --  |
| 0            | 0.40 | 0.40 | 0.40 |   5  |
| 1            | 0.40 | 0.40 | 0.40 |   5  |
| accuracy     |      |      | 0.40 |  10  |
| macro avg    | 0.40 | 0.40 | 0.40 |  10  |
| weighted avg | 0.40 | 0.40 | 0.40 |  10  |


## Logistic Regression Pipelines

### Dataset Preprocessing

```python
iris_ds = load_iris()

# train/test split
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    iris_ds.data,
    iris_ds.target,
    test_size=0.2,
    random_state=42
)
print(X_train_iris.shape, X_test_iris.shape)
# (120, 4) (30, 4)
```

### Pipeline

```python
pipe_iris = Pipeline([
    ('minmax', MinMaxScaler()),
    ('log_reg', LogisticRegression()),
])

pipe_iris.fit(X_train_iris, y_train_iris)
```

```python
iris_score = pipe_iris.score(X_test_iris, y_test_iris)
print('Prediction Accuracy: ', iris_score.round(4)*100, '%')
# Prediction Accuracy:  96.67 %
```

## Cross Validation

### Train | Test Split

#### Dataset Preprocessing

```python
!wget https://raw.githubusercontent.com/reisanar/datasets/master/Advertising.csv -P datasets
```

```python
adv_df = pd.read_csv('datasets/Advertising.csv')
adv_df.head(5)
```

|  | TV | Radio | Newspaper | Sales |
| -- | -- | -- | -- | -- |
| 0 | 230.1 | 37.8 | 69.2 | 22.1 |
| 1 | 44.5 | 39.3 | 45.1 | 10.4 |
| 2 | 17.2 | 45.9 | 69.3 | 9.3 |
| 3 | 151.5 | 41.3 | 58.5 | 18.5 |
| 4 | 180.8 | 10.8 | 58.4 | 12.9 |

```python
# Split ds into features and targets
X_adv = adv_df.drop('Sales', axis=1)
y_adv = adv_df['Sales']
```

```python
# 70:30 train/test split
X_adv_train, X_adv_test, y_adv_train, y_adv_test = train_test_split(
    X_adv, y_adv, test_size=0.3, random_state=666
)

print(X_adv_train.shape, y_adv_train.shape)
# (140, 3) (140,)
```

```python
# normalize features
scaler_adv = StandardScaler()
scaler_adv.fit(X_adv_train)

X_adv_train = scaler_adv.transform(X_adv_train)
X_adv_test = scaler_adv.transform(X_adv_test)
```

#### Model Fitting

```python
model_adv1 = Ridge(
    alpha=100.0
)

model_adv1.fit(X_adv_train, y_adv_train)
```

#### Model Evaluation

```python
y_adv_pred = model_adv1.predict(X_adv_test)

mean_squared_error(y_adv_test, y_adv_pred)
# 6.528575771818745
```

#### Adjusting Hyper Parameter

```python
model_adv2 = Ridge(
    alpha=1.0
)

model_adv2.fit(X_adv_train, y_adv_train)
```

```python
y_adv_pred2 = model_adv2.predict(X_adv_test)
mean_squared_error(y_adv_test, y_adv_pred2)
# 2.3319016551123535
```

### Train | Validation | Test Split

#### Dataset Preprocessing

```python
# 70:30 train/temp split
X_adv_train, X_adv_temp, y_adv_train, y_adv_temp = train_test_split(
    X_adv, y_adv, test_size=0.3, random_state=666
)

# 50:50 test/val split
X_adv_test, X_adv_val, y_adv_test, y_adv_val = train_test_split(
    X_adv_temp, y_adv_temp, test_size=0.5, random_state=666
)

print(X_adv_train.shape, X_adv_test.shape, X_adv_val.shape)
# (140, 3) (30, 3) (30, 3)
```

```python
# normalize features
scaler_adv = StandardScaler()
scaler_adv.fit(X_adv_train)

X_adv_train = scaler_adv.transform(X_adv_train)
X_adv_test = scaler_adv.transform(X_adv_test)
X_adv_val = scaler_adv.transform(X_adv_val)
```

#### Model Fitting and Evaluation

```python
model_adv3 = Ridge(
    alpha=100.0
)

model_adv3.fit(X_adv_train, y_adv_train)
```

```python
# do evaluation with the validation set
y_adv_pred3 = model_adv3.predict(X_adv_val)
mean_squared_error(y_adv_val, y_adv_pred3)
# 7.136230975501291
```

#### Adjusting Hyper Parameter

```python
model_adv4 = Ridge(
    alpha=1.0
)

model_adv4.fit(X_adv_train, y_adv_train)

y_adv_pred4 = model_adv4.predict(X_adv_val)
mean_squared_error(y_adv_val, y_adv_pred4)
# 2.6393803874124435
```

```python
# only once you are certain that you have the best performance
# do a final evaluation with the test set
y_adv4_final_pred = model_adv4.predict(X_adv_test)
mean_squared_error(y_adv_test, y_adv4_final_pred)
# 2.024422922812264
```

### k-fold Cross Validation

Do a train/test split and segment the training set by k-folds (e.g. 5-10) and use each of those segments once to validate a training step. The resulting error is the average of all k errors.

#### Dataset Preprocessing

```python
# 70:30 train/temp split
X_adv_train, X_adv_test, y_adv_train, y_adv_test = train_test_split(
    X_adv, y_adv, test_size=0.3, random_state=666
)
```

```python
# normalize features
scaler_adv = StandardScaler()
scaler_adv.fit(X_adv_train)

X_adv_train = scaler_adv.transform(X_adv_train)
X_adv_test = scaler_adv.transform(X_adv_test)
```

#### Model Scoring

```python
model_adv5 = Ridge(
    alpha=100.0
)
```

```python
# do a 5-fold cross-eval
scores = cross_val_score(
    estimator=model_adv5,
    X=X_adv_train,
    y=y_adv_train,
    scoring='neg_mean_squared_error',
    cv=5
)

# take the mean of all five neg. error values
abs(scores.mean())
# 8.688107513529168
```

#### Adjusting Hyper Parameter

```python
model_adv6 = Ridge(
    alpha=1.0
)
```

```python
# do a 5-fold cross-eval
scores = cross_val_score(
    estimator=model_adv6,
    X=X_adv_train,
    y=y_adv_train,
    scoring='neg_mean_squared_error',
    cv=5
)

# take the mean of all five neg. error values
abs(scores.mean())
# 3.3419582340688576
```

#### Model Fitting and Final Evaluation

```python
model_adv6.fit(X_adv_train, y_adv_train)

y_adv6_final_pred = model_adv6.predict(X_adv_test)
mean_squared_error(y_adv_test, y_adv6_final_pred)
# 2.3319016551123535
```

### Cross Validate


#### Dataset (re-import)

```python
adv_df = pd.read_csv('datasets/Advertising.csv')
X_adv = adv_df.drop('Sales', axis=1)
y_adv = adv_df['Sales']
```

```python
# 70:30 train/test split
X_adv_train, X_adv_test, y_adv_train, y_adv_test = train_test_split(
    X_adv, y_adv, test_size=0.3, random_state=666
)
```

```python
# normalize features
scaler_adv = StandardScaler()
scaler_adv.fit(X_adv_train)

X_adv_train = scaler_adv.transform(X_adv_train)
X_adv_test = scaler_adv.transform(X_adv_test)
```

#### Model Scoring

```python
model_adv7 = Ridge(
    alpha=100.0
)
```

```python
scores = cross_validate(
    model_adv7,
    X_adv_train,
    y_adv_train,
    scoring=[
        'neg_mean_squared_error',
        'neg_mean_absolute_error'
    ],
    cv=10
)
```

```python
scores_df = pd.DataFrame(scores)
scores_df
```

|   | fit_time | score_time | test_neg_mean_squared_error | test_neg_mean_absolute_error |
| -- | -- | -- | -- | -- |
| 0 | 0.016399 | 0.000749 | -12.539147 | -2.851864 |
| 1 | 0.000684 | 0.000452 | -2.806466 | -1.423516 |
| 2 | 0.000937 | 0.000782 | -11.142227 | -2.740332 |
| 3 | 0.001060 | 0.000633 | -7.237347 | -2.196963 |
| 4 | 0.001045 | 0.000738 | -11.313985 | -2.690813 |
| 5 | 0.000650 | 0.000510 | -3.169169 | -1.526568 |
| 6 | 0.000698 | 0.000429 | -6.578249 | -1.727616 |
| 7 | 0.000600 | 0.000423 | -5.740245 | -1.640964 |
| 8 | 0.000565 | 0.000463 | -10.268075 | -2.415688 |
| 9 | 0.000562 | 0.000487 | -10.641669 | -1.974407 |

```python
abs(scores_df.mean())
```

| | |
| -- | -- |
| fit_time                    |    0.002320 |
| score_time                  |    0.000566 |
| test_neg_mean_squared_error   | 8.143658 |
| test_neg_mean_absolute_error  | 2.118873 |
_dtype: float64_


#### Adjusting Hyper Parameter

```python
model_adv8 = Ridge(
    alpha=1.0
)
```

```python
scores = cross_validate(
    model_adv8,
    X_adv_train,
    y_adv_train,
    scoring=[
        'neg_mean_squared_error',
        'neg_mean_absolute_error'
    ],
    cv=10
)

abs(pd.DataFrame(scores).mean())
```

| | |
| -- | -- |
| fit_time                    |    0.001141 |
| score_time                  |    0.000777 |
| test_neg_mean_squared_error   | 3.272673 |
| test_neg_mean_absolute_error  | 1.345709 |
_dtype: float64_


#### Model Fitting and Final Evaluation

```python
model_adv8.fit(X_adv_train, y_adv_train)

y_adv8_final_pred = model_adv8.predict(X_adv_test)
mean_squared_error(y_adv_test, y_adv8_final_pred)
# 2.3319016551123535
```

### Grid Search

Loop through a set of hyperparameters to find an optimum.


#### Hyperparameter Search

```python
base_elastic_net_model = ElasticNet()
```

```python
param_grid = {
    'alpha': [0.1, 1, 5, 10, 50, 100],
    'l1_ratio':[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
}
```

```python
grid_model = GridSearchCV(
    estimator=base_elastic_net_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5, verbose=2
)

grid_model.fit(X_adv_train, y_adv_train)

print(
    'Results:\nBest Estimator: ',
    grid_model.best_estimator_,
    '\nBest Hyperparameter: ',
    grid_model.best_params_
)
```

__Results__:
* Best Estimator:  `ElasticNet(alpha=0.1, l1_ratio=1.0)`
* Best Hyperparameter:  `{'alpha': 0.1, 'l1_ratio': 1.0}`

```python
gridcv_results = pd.DataFrame(grid_model.cv_results_)
```

|  | mean_fit_time | std_fit_time | mean_score_time | std_score_time | param_alpha | param_l1_ratio | params | split0_test_score | split1_test_score | split2_test_score | split3_test_score | split4_test_score | mean_test_score | std_test_score | rank_test_score |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 0 | 0.001156 | 0.000160 | 0.000449 | 0.000038 | 0.1 | 0.1 | {'alpha': 0.1, 'l1_ratio': 0.1} | -1.924119 | -3.384152 | -3.588444 | -3.703040 | -5.091974 | -3.538346 | 1.007264 | 6 |
| 1 | 0.001144 | 0.000181 | 0.000407 | 0.000091 | 0.1 | 0.3 | {'alpha': 0.1, 'l1_ratio': 0.3} | -1.867117 | -3.304382 | -3.561106 | -3.623188 | -5.061781 | -3.483515 | 1.016000 | 5 |
| 2 | 0.000623 | 0.000026 | 0.000272 | 0.000052 | 0.1 | 0.5 | {'alpha': 0.1, 'l1_ratio': 0.5} | -1.812633 | -3.220727 | -3.539711 | -3.547572 | -5.043259 | -3.432780 | 1.028406 | 4 |
| 3 | 0.000932 | 0.000165 | 0.000321 | 0.000060 | 0.1 | 0.7 | {'alpha': 0.1, 'l1_ratio': 0.7} | -1.750153 | -3.144120 | -3.525226 | -3.477228 | -5.034008 | -3.386147 | 1.046722 | 3 |
| 4 | 0.000725 | 0.000106 | 0.000259 | 0.000024 | 0.1 | 0.9 | {'alpha': 0.1, 'l1_ratio': 0.9} | -1.693440 | -3.075686 | -3.518777 | -3.413393 | -5.029683 | -3.346196 | 1.065195 | 2 |
| 5 | 0.000654 | 0.000053 | 0.000274 | 0.000026 | 0.1 | 1.0 | {'alpha': 0.1, 'l1_ratio': 1.0} | -1.667506 | -3.044928 | -3.518866 | -3.384363 | -5.031297 | -3.329392 | 1.075006 | 1 |
| 6 | 0.000595 | 0.000016 | 0.000244 | 0.000002 | 1 | 0.1 | {'alpha': 1, 'l1_ratio': 0.1} | -8.575470 | -11.021534 | -8.212152 | -6.808719 | -10.792072 | -9.081990 | 1.604192 | 12 |
| 7 | 0.000591 | 0.000018 | 0.000244 | 0.000002 | 1 | 0.3 | {'alpha': 1, 'l1_ratio': 0.3} | -8.131855 | -10.448423 | -7.774620 | -6.179358 | -10.071728 | -8.521197 | 1.569173 | 11 |
| 8 | 0.000628 | 0.000049 | 0.000266 | 0.000023 | 1 | 0.5 | {'alpha': 1, 'l1_ratio': 0.5} | -7.519809 | -9.562473 | -7.261824 | -5.453399 | -9.213320 | -7.802165 | 1.481785 | 10 |
| 9 | 0.000594 | 0.000015 | 0.000243 | 0.000002 | 1 | 0.7 | {'alpha': 1, 'l1_ratio': 0.7} | -6.614835 | -8.351711 | -6.702104 | -4.698977 | -8.230616 | -6.919649 | 1.329741 | 9 |
| 10 | 0.000714 | 0.000108 | 0.000268 | 0.000033 | 1 | 0.9 | {'alpha': 1, 'l1_ratio': 0.9} | -5.537250 | -6.887828 | -6.148400 | -4.106124 | -7.101573 | -5.956235 | 1.078430 | 8 |
| 11 | 0.000649 | 0.000067 | 0.000263 | 0.000028 | 1 | 1.0 | {'alpha': 1, 'l1_ratio': 1.0} | -4.932027 | -6.058207 | -5.892529 | -3.798441 | -6.472871 | -5.430815 | 0.959804 | 7 |
| 12 | 0.000645 | 0.000042 | 0.000264 | 0.000040 | 5 | 0.1 | {'alpha': 5, 'l1_ratio': 0.1} | -21.863798 | -25.767488 | -18.768865 | -12.608680 | -23.207907 | -20.443347 | 4.520904 | 13 |
| 13 | 0.000617 | 0.000030 | 0.000281 | 0.000038 | 5 | 0.3 | {'alpha': 5, 'l1_ratio': 0.3} | -23.626694 | -27.439028 | -20.266203 | -12.788078 | -24.609195 | -21.745840 | 5.031493 | 14 |
| 14 | 0.000599 | 0.000011 | 0.000249 | 0.000013 | 5 | 0.5 | {'alpha': 5, 'l1_ratio': 0.5} | -26.202964 | -29.867138 | -22.527913 | -13.423857 | -26.835934 | -23.771561 | 5.675911 | 15 |
| 15 | 0.000588 | 0.000013 | 0.000276 | 0.000035 | 5 | 0.7 | {'alpha': 5, 'l1_ratio': 0.7} | -27.768946 | -33.428462 | -23.506474 | -14.599984 | -29.112276 | -25.683228 | 6.382379 | 17 |
| 16 | 0.000580 | 0.000003 | 0.000271 | 0.000001 | 5 | 0.9 | {'alpha': 5, 'l1_ratio': 0.9} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 17 | 0.000591 | 0.000011 | 0.000259 | 0.000021 | 5 | 1.0 | {'alpha': 5, 'l1_ratio': 1.0} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 18 | 0.000632 | 0.000028 | 0.000250 | 0.000012 | 10 | 0.1 | {'alpha': 10, 'l1_ratio': 0.1} | -26.179546 | -30.396420 | -22.386698 | -14.596498 | -27.292337 | -24.170300 | 5.429322 | 16 |
| 19 | 0.000593 | 0.000020 | 0.000239 | 0.000001 | 10 | 0.3 | {'alpha': 10, 'l1_ratio': 0.3} | -28.704426 | -33.379967 | -24.561645 | -15.634153 | -29.883725 | -26.432783 | 6.090062 | 18 |
| 20 | 0.000595 | 0.000036 | 0.000245 | 0.000013 | 10 | 0.5 | {'alpha': 10, 'l1_ratio': 0.5} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 21 | 0.000610 | 0.000053 | 0.000258 | 0.000015 | 10 | 0.7 | {'alpha': 10, 'l1_ratio': 0.7} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 22 | 0.000597 | 0.000022 | 0.000248 | 0.000015 | 10 | 0.9 | {'alpha': 10, 'l1_ratio': 0.9} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 23 | 0.000623 | 0.000057 | 0.000305 | 0.000076 | 10 | 1.0 | {'alpha': 10, 'l1_ratio': 1.0} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 24 | 0.000602 | 0.000016 | 0.000252 | 0.000013 | 50 | 0.1 | {'alpha': 50, 'l1_ratio': 0.1} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 25 | 0.000577 | 0.000009 | 0.000238 | 0.000001 | 50 | 0.3 | {'alpha': 50, 'l1_ratio': 0.3} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 26 | 0.000607 | 0.000046 | 0.000245 | 0.000010 | 50 | 0.5 | {'alpha': 50, 'l1_ratio': 0.5} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 27 | 0.000569 | 0.000004 | 0.000259 | 0.000012 | 50 | 0.7 | {'alpha': 50, 'l1_ratio': 0.7} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 28 | 0.000582 | 0.000022 | 0.000244 | 0.000011 | 50 | 0.9 | {'alpha': 50, 'l1_ratio': 0.9} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 29 | 0.000603 | 0.000041 | 0.000251 | 0.000015 | 50 | 1.0 | {'alpha': 50, 'l1_ratio': 1.0} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 30 | 0.000670 | 0.000106 | 0.000251 | 0.000013 | 100 | 0.1 | {'alpha': 100, 'l1_ratio': 0.1} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 31 | 0.000764 | 0.000179 | 0.000343 | 0.000054 | 100 | 0.3 | {'alpha': 100, 'l1_ratio': 0.3} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 32 | 0.000623 | 0.000077 | 0.000244 | 0.000007 | 100 | 0.5 | {'alpha': 100, 'l1_ratio': 0.5} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 33 | 0.000817 | 0.000156 | 0.000329 | 0.000076 | 100 | 0.7 | {'alpha': 100, 'l1_ratio': 0.7} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 34 | 0.000590 | 0.000017 | 0.000242 | 0.000004 | 100 | 0.9 | {'alpha': 100, 'l1_ratio': 0.9} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |
| 35 | 0.000595 | 0.000027 | 0.000242 | 0.000007 | 100 | 1.0 | {'alpha': 100, 'l1_ratio': 1.0} | -29.868949 | -34.423737 | -25.623955 | -16.750237 | -31.056181 | -27.544612 | 6.087093 | 19 |

```python
gridcv_results[
    [
        'param_alpha',
        'param_l1_ratio'
    ]
].plot(title='Grid Search Hyperparameter :: Parameter', figsize=(12,8))
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_08.webp)

```python
gridcv_results[
    [
        'mean_fit_time',
        'std_fit_time',
        'mean_score_time'
    ]
].plot(title='Grid Search Hyperparameter :: Timing', figsize=(12,8))
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_09.webp)

```python
gridcv_results[
    [
        'split0_test_score',
        'split1_test_score',
        'split2_test_score',
        'split3_test_score',
        'split4_test_score',
        'mean_test_score',
        'std_test_score',
       'rank_test_score'
    ]
].plot(title='Grid Search Hyperparameter :: Parameter', figsize=(12,8))
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_10.webp)


#### Model Evaluation

```python
y_grid_pred = grid_model.predict(X_adv_test)

mean_squared_error(y_adv_test, y_grid_pred)
# 2.380865536033581
```

## KNN Algorithm

### Dataset

```python
wine = load_wine()
print(wine.data.shape)
print(wine.feature_names)
print(wine.data[:1])

# (178, 13)
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
# [[1.423e+01 1.710e+00 2.430e+00 1.560e+01 1.270e+02 2.800e+00 3.060e+00
#   2.800e-01 2.290e+00 5.640e+00 1.040e+00 3.920e+00 1.065e+03]]
```

```python
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
wine_df.head(2).T
```

|  | 0 | 1 |
| -- | -- | -- |
| alcohol | 14.23 | 13.20 |
| malic_acid | 1.71 | 1.78 |
| ash | 2.43 | 2.14 |
| alcalinity_of_ash | 15.60 | 11.20 |
| magnesium | 127.00 | 100.00 |
| total_phenols | 2.80 | 2.65 |
| flavanoids | 3.06 | 2.76 |
| nonflavanoid_phenols | 0.28 | 0.26 |
| proanthocyanins | 2.29 | 1.28 |
| color_intensity | 5.64 | 4.38 |
| hue | 1.04 | 1.05 |
| od280/od315_of_diluted_wines | 3.92 | 3.40 |
| proline | 1065.00 | 1050.00 |


### Data Pre-processing

```python
# normalization
scaler = MinMaxScaler()
scaler.fit(wine.data)
wine_norm = scaler.fit_transform(wine.data)
```

```python
# train/test split
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
    wine_norm,
    wine.target,
    test_size=0.3
)

print(X_train_wine.shape, X_test_wine.shape)
# (124, 13) (54, 13)
```

### Model Fitting

```python
# model for k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_wine, y_train_wine)

y_pred_wine_knn3 = knn.predict(X_test_wine)
print('Accuracy Score: ', (accuracy_score(y_test_wine, y_pred_wine_knn3)*100).round(2), '%')
# Accuracy Score:  98.15 %
```

```python
# model for k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_wine, y_train_wine)

y_pred_wine_knn5 = knn.predict(X_test_wine)
print('Accuracy Score: ', (accuracy_score(y_test_wine, y_pred_wine_knn5)*100).round(2), '%')
# Accuracy Score:  98.15 %
```

```python
# model for k=7
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_wine, y_train_wine)

y_pred_wine_knn7 = knn.predict(X_test_wine)
print('Accuracy Score: ', (accuracy_score(y_test_wine, y_pred_wine_knn7)*100).round(2), '%')
# Accuracy Score:  96.3 %
```

```python
# model for k=9
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_wine, y_train_wine)

y_pred_wine_knn7 = knn.predict(X_test_wine)
print('Accuracy Score: ', (accuracy_score(y_test_wine, y_pred_wine_knn7)*100).round(2), '%')
# Accuracy Score:  96.3 %
```

## Decision Tree Classifier

* Does not require normalization
* Is not sensitive to missing values

### Dataset

```python
!wget https://gist.githubusercontent.com/Dviejopomata/ea5869ba4dcff84f8c294dc7402cd4a9/raw/4671f90b8b04ba4db9d67acafaa4c0827cd233c2/bill_authentication.csv -P datasets
```

```python
bill_auth_df = pd.read_csv('datasets/bill_authentication.csv')
bill_auth_df.head(3)
```

|   | Variance | Skewness | Curtosis | Entropy | Class |
|  -- | -- | -- | -- | -- | -- |
| 0 | 3.6216 | 8.6661 | -2.8073 | -0.44699 | 0 |
| 1 | 4.5459 | 8.1674 | -2.4586 | -1.46210 | 0 |
| 2 | 3.8660 | -2.6383 | 1.9242 | 0.10645 | 0 |


### Preprocessing

```python
# remove target feature from training set
X_bill = bill_auth_df.drop('Class', axis=1)
y_bill = bill_auth_df['Class']
```

```python
X_train_bill, X_test_bill, y_train_bill, y_test_bill = train_test_split(X_bill, y_bill, test_size=0.2)
```

### Model Fitting

```python
tree_classifier = DecisionTreeClassifier()

tree_classifier.fit(X_train_bill, y_train_bill)
```

### Evaluation

```python
y_pred_bill = tree_classifier.predict(X_test_bill)
```

```python
conf_mtx_bill = confusion_matrix(y_test_bill, y_pred_bill)
conf_mtx_bill

# array([[150,   2],
#        [  4, 119]])
```

```python
conf_mtx_bill_plot = ConfusionMatrixDisplay(
    confusion_matrix=conf_mtx_bill,
    display_labels=[False,True]
)

conf_mtx_bill_plot.plot()
plt.show()
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_05.webp)

```python
report_bill = classification_report(
    y_test_bill, y_pred_bill
)
print(report_bill)
```

|              | precision | recall | f1-score | support |
|    --        |  --  |  --  |  --  |  --  |
| 0            | 0.97 | 0.99 | 0.98 |   152  |
| 1            | 0.98 | 0.97 | 0.98 |   123  |
| accuracy     |      |      | 0.98 |  275  |
| macro avg    | 0.98 | 0.98 | 0.98 |  275  |
| weighted avg | 0.98 | 0.98 | 0.98 |  275  |


## Random Forest Classifier

* Does not require normalization
* Is not sensitive to missing values
* Low risk of overfitting
* Efficient with large datasets
* High accuracy

### Dataset

```python
!wget https://raw.githubusercontent.com/xjcjiacheng/data-analysis/master/heart%20disease%20UCI/heart.csv -P datasets
```

```python
heart_df = pd.read_csv('datasets/heart.csv')
heart_df.head(5)
```

|   | age | sex | cp | trestbps | chol | fbs | restecg | thalach | exang | oldpeak | slope | ca | thal | target |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 0 | 63 | 1 | 3 | 145 | 233 | 1 | 0 | 150 | 0 | 2.3 | 0 | 0 | 1 | 1 |
| 1 | 37 | 1 | 2 | 130 | 250 | 0 | 1 | 187 | 0 | 3.5 | 0 | 0 | 2 | 1 |
| 2 | 41 | 0 | 1 | 130 | 204 | 0 | 0 | 172 | 0 | 1.4 | 2 | 0 | 2 | 1 |
| 3 | 56 | 1 | 1 | 120 | 236 | 0 | 1 | 178 | 0 | 0.8 | 2 | 0 | 2 | 1 |
| 4 | 57 | 0 | 0 | 120 | 354 | 0 | 1 | 163 | 1 | 0.6 | 2 | 0 | 2 | 1 |


### Preprocessing

```python
# remove target feature from training set
X_heart = heart_df.drop('target', axis=1)
y_heart = heart_df['target']
```

```python
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(
    X_heart,
    y_heart,
    test_size=0.2,
    random_state=0
)
```

### Model Fitting

```python
forest_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')

forest_classifier.fit(X_train_heart, y_train_heart)
```

### Evaluation

```python
y_pred_heart = forest_classifier.predict(X_test_heart)
```

```python
conf_mtx_heart = confusion_matrix(y_test_heart, y_pred_heart)
conf_mtx_heart

# array([[24,  3],
#        [ 5, 29]])
```

```python
conf_mtx_heart_plot = ConfusionMatrixDisplay(
    confusion_matrix=conf_mtx_heart,
    display_labels=[False,True]
)

conf_mtx_heart_plot.plot()
plt.show()
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_06.webp)

```python
report_heart = classification_report(
    y_test_heart, y_pred_heart
)
print(report_heart)
```

|              | precision | recall | f1-score | support |
|    --        |  --  |  --  |  --  |  --  |
| 0            | 0.83 | 0.89 | 0.86 |   27  |
| 1            | 0.91 | 0.85 | 0.88 |   34  |
| accuracy     |      |      | 0.87 |  61  |
| macro avg    | 0.87 | 0.87 | 0.87 |  61  |
| weighted avg | 0.87 | 0.87 | 0.87 |  61  |


## Random Forest Hyperparameter Tuning


### Testing Hyperparameters

```python
rdnfor_classifier = RandomForestClassifier(
    n_estimators=2,
    min_samples_split=2,
    min_samples_leaf=1,
    criterion='entropy'
)
rdnfor_classifier.fit(X_train_heart, y_train_heart)
```

```python
rdnfor_pred = rdnfor_classifier.predict(X_test_heart)
print('Accuracy Score: ', accuracy_score(y_test_heart, rdnfor_pred).round(4)*100, '%')

# Accuracy Score:  73.77 %
```

### Grid-Search Cross-Validation

Try a set of values for selected Hyperparameter to find the optimal configuration.

```python
param_grid = {
    'n_estimators': [5, 25, 50, 75,100, 125],
    'min_samples_split': [1,2,3],
    'min_samples_leaf': [1,2,3],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_features' : ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator = rdnfor_classifier,
    param_grid = param_grid
)

grid_search.fit(X_train_heart, y_train_heart)
```

```python
print('Best Parameter: ', grid_search.best_params_)
# Best Parameter:  {
# 'criterion': 'entropy',
# 'max_features': 'sqrt',
# 'min_samples_leaf': 2,
# 'min_samples_split': 1,
# 'n_estimators': 25
# }
```

```python
rdnfor_classifier_optimized = RandomForestClassifier(
    n_estimators=25,
    min_samples_split=1,
    min_samples_leaf=2,
    criterion='entropy',
    max_features='sqrt'
)

rdnfor_classifier_optimized.fit(X_train_heart, y_train_heart)
```

```python
rdnfor_pred_optimized = rdnfor_classifier_optimized.predict(X_test_heart)
print('Accuracy Score: ', accuracy_score(y_test_heart, rdnfor_pred_optimized).round(4)*100, '%')

# Accuracy Score:  85.25 %
```

### Random Forest Classifier 1 - Penguins

```python
!wget https://github.com/remijul/dataset/raw/master/penguins_size.csv -P datasets
```

```python
peng_df = pd.read_csv('datasets/penguins_size.csv')
peng_df = peng_df.dropna()
peng_df.head(5)
```

|   | species | island | culmen_length_mm | culmen_depth_mm | flipper_length_mm | body_mass_g | sex |
| -- | -- | -- | -- | -- | -- | -- | -- |
| 0 | Adelie | Torgersen | 39.1 | 18.7 | 181.0 | 3750.0 | MALE |
| 1 | Adelie | Torgersen | 39.5 | 17.4 | 186.0 | 3800.0 | FEMALE |
| 2 | Adelie | Torgersen | 40.3 | 18.0 | 195.0 | 3250.0 | FEMALE |
| 4 | Adelie | Torgersen | 36.7 | 19.3 | 193.0 | 3450.0 | FEMALE |
| 5 | Adelie | Torgersen | 39.3 | 20.6 | 190.0 | 3650.0 | MALE |

```python
# drop labels and encode string values
X_peng = pd.get_dummies(peng_df.drop('species', axis=1),drop_first=True)
y_peng = peng_df['species']
```

```python
# train/test split
X_peng_train, X_peng_test, y_peng_train, y_peng_test = train_test_split(
    X_peng,
    y_peng,
    test_size=0.3,
    random_state=42
)
```

```python
# creating the model
rfc_peng = RandomForestClassifier(
    n_estimators=10,
    max_features='sqrt',
    random_state=42
)
```

```python
# model training and running predictions
rfc_peng.fit(X_peng_train, y_peng_train)
peng_pred = rfc_peng.predict(X_peng_test)
print('Accuracy Score: ',accuracy_score(y_peng_test, peng_pred, normalize=True).round(4)*100, '%')
# Accuracy Score:  98.02 %
```

#### Feature Importance

```python
# feature importance for classification
peng_index = ['importance']
peng_data_columns = pd.Series(X_peng.columns)
peng_importance_array = rfc_peng.feature_importances_
peng_importance_df = pd.DataFrame(peng_importance_array, peng_data_columns, peng_index)
peng_importance_df
```

|  | importance |
| -- | -- |
| culmen_length_mm | 0.288928 |
| culmen_depth_mm | 0.111021 |
| flipper_length_mm | 0.357994 |
| body_mass_g | 0.025477 |
| island_Dream | 0.178498 |
| island_Torgersen | 0.031042 |
| sex_FEMALE | 0.004716 |
| sex_MALE | 0.002324 |

```python
peng_importance_df.sort_values(
    by='importance',
    ascending=False
).plot(
    kind='barh',
    title='Feature Importance for Species Classification',
    figsize=(12,4)
)
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_28.webp)


#### Model Evaluation

```python
report_peng = classification_report(y_peng_test, peng_pred)
print(report_peng)
```

|  | precision | recall | f1-score | support |
| -- | -- | -- | -- | -- |
| Adelie | 0.98 | 0.98 | 0.98 | 49 |
| Chinstrap | 0.94 | 0.94 | 0.94 | 18 |
| Gentoo | 1.00 | 1.00 | 1.00 |  34 |
|     accuracy |      |      | 0.98 | 101 |
|    macro avg | 0.97 | 0.97 | 0.97 | 101 |
| weighted avg | 0.98 | 0.98 | 0.98 | 101 |

```python
conf_mtx_peng = confusion_matrix(y_peng_test, peng_pred)

conf_mtx_peng_plot = ConfusionMatrixDisplay(
    confusion_matrix=conf_mtx_peng
)

conf_mtx_peng_plot.plot(cmap='plasma')
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_27.webp)


### Random Forest Classifier - Banknote Authentication

```python
!wget https://github.com/jbrownlee/Datasets/raw/master/banknote_authentication.csv -P datasets
```

```python
money_df = pd.read_csv('datasets/data-banknote-authentication.csv')
money_df.head(5)
```

|  | Variance_Wavelet | Skewness_Wavelet | Curtosis_Wavelet | Image_Entropy | Class |
| -- | -- | -- | -- | -- | -- |
| 0 | 3.62160 | 8.6661 | -2.8073 | -0.44699 | 0 |
| 1 | 4.54590 | 8.1674 | -2.4586 | -1.46210 | 0 |
| 2 | 3.86600 | -2.6383 | 1.9242 | 0.10645 | 0 |
| 3 | 3.45660 | 9.5228 | -4.0112 | -3.59440 | 0 |
| 4 | 0.32924 | -4.4552 | 4.5718 | -0.98880 | 0 |

```python
sns.pairplot(money_df, hue='Class', palette='winter')
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_29.webp)

```python
# drop label for training
X_money = money_df.drop('Class', axis=1)
y_money = money_df['Class']
print(X_money.shape, y_money.shape)
```

```python
X_money_train, X_money_test, y_money_train, y_money_test = train_test_split(
    X_money,
    y_money,
    test_size=0.15,
    random_state=42
)
```

#### Grid Search for Hyperparameters

```python
rfc_money_base = RandomForestClassifier(oob_score=True)
```

```python
param_grid = {
    'n_estimators': [64, 96, 128, 160, 192],
    'max_features': [2,3,4],
    'bootstrap': [True, False]
}
```

```python
grid_money = GridSearchCV(rfc_money_base, param_grid) 
grid_money.fit(X_money_train, y_money_train)
grid_money.best_params_
# {'bootstrap': True, 'max_features': 2, 'n_estimators': 96}
```

#### Model Training and Evaluation

```python
rfc_money = RandomForestClassifier(
    bootstrap=True,
    max_features=2,
    n_estimators=96,
    oob_score=True
)
rfc_money.fit(X_money_train, y_money_train)
print('Out-of-Bag Score: ', rfc_money.oob_score_.round(4)*100, '%')
# Out-of-Bag Score:  99.14 %
```

```python
money_pred = rfc_money.predict(X_money_test)
money_report = classification_report(y_money_test, money_pred)
print(money_report)
```

|  | precision | recall | f1-score | support |
| -- | -- | -- | -- | -- |
|     0 | 0.99 | 1.00 | 1.00 | 111 |
|     1 | 1.00 | 0.99 | 0.99 |  95 |
|    accuracy |  |  | 1.00 | 206 |
|   macro avg | 1.00 | 0.99 | 1.00 | 206 |
|weighted avg | 1.00 | 1.00 | 1.00 | 206 |

```python
conf_mtx_money = confusion_matrix(y_money_test, money_pred)

conf_mtx_money_plot = ConfusionMatrixDisplay(
    confusion_matrix=conf_mtx_money
)

conf_mtx_money_plot.plot(cmap='plasma')
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_30.webp)


#### Optimizations

```python
# verify number of estimators found by grid search
errors = []
missclassifications = []

for n in range(1,200):
    rfc = RandomForestClassifier(n_estimators=n, max_features=2)
    rfc.fit(X_money_train, y_money_train)
    preds = rfc.predict(X_money_test)
    
    err = 1 - accuracy_score(y_money_test, preds)
    errors.append(err)
    
    n_missed = np.sum(preds != y_money_test)
    missclassifications.append(n_missed)
```

```python
plt.figure(figsize=(12,4))
plt.title('Errors as a Function of n_estimators')
plt.xlabel('Estimators')
plt.ylabel('Error Score')
plt.plot(range(1,200), errors)
# there is no noteable improvement above ~10 estimators
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_31.webp)

```python
plt.figure(figsize=(12,4))
plt.title('Misclassifications as a Function of n_estimators')
plt.xlabel('Estimators')
plt.ylabel('Misclassifications')
plt.plot(range(1,200), missclassifications)
# and the same for misclassifications
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_32.webp)


## Random Forest Regressor

Comparing different regression models to a random forrest regression model.

```python
# dataset
!wget https://github.com/vineetsingh028/Rock_Density_Prediction/raw/master/rock_density_xray.csv -P datasets
```

```python
rock_df = pd.read_csv('datasets/rock_density_xray.csv')
rock_df.columns = ['Signal', 'Density']
rock_df.head(5)
```

|   | Signal | Density |
| -- | -- | -- |
| 0 | 72.945124 | 2.456548 |
| 1 | 14.229877 | 2.601719 |
| 2 | 36.597334 | 1.967004 |
| 3 | 9.578899 | 2.300439 |
| 4 | 21.765897 | 2.452374 |

```python
plt.figure(figsize=(12,5))
plt.title('X-Ray Bounce Signal Strength vs Rock Density')
sns.scatterplot(data=rock_df, x='Signal', y='Density')
# the signal vs density plot follows a sine wave - spoiler alert: simpler algorithm
# will fail trying to fit this dataset...
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_33.webp)

```python
# train-test split
X_rock = rock_df['Signal'].values.reshape(-1,1)
y_rock = rock_df['Density']

X_rock_train, X_rock_test, y_rock_train, y_rock_test = train_test_split(
    X_rock,
    y_rock,
    test_size=0.1,
    random_state=42
)
```

```python
# normalization
scaler = StandardScaler()
X_rock_train_scaled = scaler.fit_transform(X_rock_train)
X_rock_test_scaled = scaler.transform(X_rock_test)
```

### vs Linear Regression

```python
lr_rock = LinearRegression()
lr_rock.fit(X_rock_train_scaled, y_rock_train)
```

```python
lr_rock_preds = lr_rock.predict(X_rock_test_scaled)

mae = mean_absolute_error(y_rock_test, lr_rock_preds)
rmse = np.sqrt(mean_squared_error(y_rock_test, lr_rock_preds))
mean_abs = y_rock_test.mean()
avg_error = mae * 100 / mean_abs

print('MAE: ', mae.round(2), 'RMSE: ', rmse.round(2), 'Relative Avg. Error: ', avg_error.round(2), '%')
# MAE:  0.24 RMSE:  0.3 Relative Avg. Error:  10.93 %
```

```python
# visualize predictions
plt.figure(figsize=(12,5))
plt.plot(X_rock_test, lr_rock_preds, c='mediumspringgreen')
sns.scatterplot(data=rock_df, x='Signal', y='Density', c='dodgerblue')
plt.title('Linear Regression Predictions')
plt.show()
# the returned error appears small because the linear regression returns an average
# but it cannot fit a linear line to the contours of the underlying sine wave function
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_34.webp)


### vs Polynomial Regression

```python
# helper function
def run_model(model, X_train, y_train, X_test, y_test, df):
    
    # FIT MODEL
    model.fit(X_train, y_train)
    
    # EVALUATE
    y_preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_preds)
    rmse = np.sqrt(mean_squared_error(y_test, y_preds))
    mean_abs = y_test.mean()
    avg_error = mae * 100 / mean_abs
    print('MAE: ', mae.round(2), 'RMSE: ', rmse.round(2), 'Relative Avg. Error: ', avg_error.round(2), '%')
    
    # PLOT RESULTS
    signal_range = np.arange(0,100)
    output = model.predict(signal_range.reshape(-1,1))
    
    
    plt.figure(figsize=(12,5))
    sns.scatterplot(data=df, x='Signal', y='Density', c='dodgerblue')
    plt.plot(signal_range,output, c='mediumspringgreen')
    plt.title('Regression Predictions')
    plt.show()
```

```python
# test helper on previous linear regression
run_model(
    model=lr_rock,
    X_train=X_rock_train,
    y_train=y_rock_train,
    X_test=X_rock_test,
    y_test=y_rock_test,
    df=rock_df
)
```

> MAE:  0.24 RMSE:  0.3 Relative Avg. Error:  10.93 %

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_35.webp)

```python
# build polynomial model
pipe_poly = make_pipeline(
    PolynomialFeatures(degree=6),
    LinearRegression()
)
```

```python
# run model
run_model(
    model=pipe_poly,
    X_train=X_rock_train,
    y_train=y_rock_train,
    X_test=X_rock_test,
    y_test=y_rock_test,
    df=rock_df
)
# with a HARD LIMIT of 0-100 for the xray signal a 6th degree polinomial is a good fit
```

> MAE:  0.13 RMSE:  0.14 Relative Avg. Error:  5.7 %

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_36.webp)


### vs KNeighbors Regression

```python
# build polynomial model
k_values=[1,5,10,25]

for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k)
    print(model)
    
    # run model
    run_model(
        model,
        X_train=X_rock_train,
        y_train=y_rock_train,
        X_test=X_rock_test,
        y_test=y_rock_test,
        df=rock_df
    )
```
> KNeighborsRegressor(n_neighbors=1)
>
> MAE:  0.12 RMSE:  0.17 Relative Avg. Error:  5.47 %

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_37.webp)

> KNeighborsRegressor()
>
> MAE:  0.13 RMSE:  0.15 Relative Avg. Error:  5.9 %

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_38.webp)

> KNeighborsRegressor(n_neighbors=10)
>
> MAE:  0.12 RMSE:  0.14 Relative Avg. Error:  5.44 %

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_39.webp)

> KNeighborsRegressor(n_neighbors=25)
>
> MAE:  0.14 RMSE:  0.16 Relative Avg. Error:  6.18 %

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_40.webp)


### vs Decision Tree Regression

```python
tree_model = DecisionTreeRegressor()

# run model
run_model(
    model=tree_model,
    X_train=X_rock_train,
    y_train=y_rock_train,
    X_test=X_rock_test,
    y_test=y_rock_test,
    df=rock_df
)
```

> MAE:  0.12 RMSE:  0.17 Relative Avg. Error:  5.47 %

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_41.webp)


### vs Support Vector Regression

```python
svr_rock = svm.SVR()

param_grid = {
    'C': [0.01,0.1,1,5,10,100, 1000],
    'gamma': ['auto', 'scale']
}

rock_grid = GridSearchCV(svr_rock, param_grid)
```

```python
# run model
run_model(
    model=rock_grid,
    X_train=X_rock_train,
    y_train=y_rock_train,
    X_test=X_rock_test,
    y_test=y_rock_test,
    df=rock_df
)
```

> MAE:  0.13 RMSE:  0.14 Relative Avg. Error:  5.75 %

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_42.webp)


### vs Gradient Boosting Regression

```python
gbr_rock = GradientBoostingRegressor()

# run model
run_model(
    model=gbr_rock,
    X_train=X_rock_train,
    y_train=y_rock_train,
    X_test=X_rock_test,
    y_test=y_rock_test,
    df=rock_df
)
```

> MAE:  0.13 RMSE:  0.15 Relative Avg. Error:  5.76 %

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_44.webp)


### vs Ada Boosting Regression

```python
abr_rock = AdaBoostRegressor()

# run model
run_model(
    model=abr_rock,
    X_train=X_rock_train,
    y_train=y_rock_train,
    X_test=X_rock_test,
    y_test=y_rock_test,
    df=rock_df
)
```

> MAE:  0.13 RMSE:  0.14 Relative Avg. Error:  5.67 %

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_45.webp)


### Finally, Random Forrest Regression

```python
rfr_rock = RandomForestRegressor(n_estimators=10)

# run model
run_model(
    model=rfr_rock,
    X_train=X_rock_train,
    y_train=y_rock_train,
    X_test=X_rock_test,
    y_test=y_rock_test,
    df=rock_df
)
```

> MAE:  0.11 RMSE:  0.14 Relative Avg. Error:  5.1 %

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_43.webp)

<!-- #region -->
## SVC Model

__Support Vector Machines__ (`SVM`s) are a set of supervised learning methods used for classification, regression and outliers detection.


* Effective in high dimensional spaces.
* Still effective in cases where number of dimensions is greater than the number of samples.
<!-- #endregion -->

### Dataset

* [Three different varieties of the wheat - Kaggle.com](https://www.kaggle.com/datasets/dongeorge/seed-from-uci)

Measurements of geometrical properties of kernels belonging to three different varieties of wheat:

* __A__: Area,
* __P__: Perimeter,
* __C__ = 4piA/P^2: Compactness,
* __LK__: Length of kernel,
* __WK__: Width of kernel,
* __A\_Coef__: Asymmetry coefficient
* __LKG__: Length of kernel groove.

```python
!wget https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/Seed_Data.csv -P datasets
```

```python
wheat_df = pd.read_csv('datasets/Seed_Data.csv')
wheat_df.head(5)
```

|   | A | P | C | LK | WK | A_Coef | LKG | target |
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 0 | 15.26 | 14.84 | 0.8710 | 5.763 | 3.312 | 2.221 | 5.220 | 0 |
| 1 | 14.88 | 14.57 | 0.8811 | 5.554 | 3.333 | 1.018 | 4.956 | 0 |
| 2 | 14.29 | 14.09 | 0.9050 | 5.291 | 3.337 | 2.699 | 4.825 | 0 |
| 3 | 13.84 | 13.94 | 0.8955 | 5.324 | 3.379 | 2.259 | 4.805 | 0 |
| 4 | 16.14 | 14.99 | 0.9034 | 5.658 | 3.562 | 1.355 | 5.175 | 0 |

```python
wheat_df.info()

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 210 entries, 0 to 209
# Data columns (total 8 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   A       210 non-null    float64
#  1   P       210 non-null    float64
#  2   C       210 non-null    float64
#  3   LK      210 non-null    float64
#  4   WK      210 non-null    float64
#  5   A_Coef  210 non-null    float64
#  6   LKG     210 non-null    float64
#  7   target  210 non-null    int64  
# dtypes: float64(7), int64(1)
# memory usage: 13.2 KB
```

#### Preprocessing

```python
# remove target feature from training set
X_wheat = wheat_df.drop('target', axis=1)
y_wheat = wheat_df['target']

print(X_wheat.shape, y_wheat.shape)
# (210, 7) (210,)
```

```python
# train/test split
X_train_wheat, X_test_wheat, y_train_wheat, y_test_wheat = train_test_split(
    X_wheat,
    y_wheat,
    test_size=0.2,
    random_state=42
)
```

```python
# normalization
sc_wheat = StandardScaler()
X_train_wheat=sc_wheat.fit_transform(X_train_wheat)
X_test_wheat=sc_wheat.fit_transform(X_test_wheat)
```

#### Model Training

```python
# SVM classifier fitting
clf_wheat = svm.SVC()
clf_wheat.fit(X_train_wheat, y_train_wheat)
```

#### Model Evaluation

```python
# Predictions
y_wheat_pred = clf_wheat.predict(X_test_wheat)
```

```python
print(
    'Accuracy Score: ',
    accuracy_score(y_test_wheat, y_wheat_pred, normalize=True).round(4)*100, '%'
)
# Accuracy Score:  90.48 %
```

```python
report_wheat = classification_report(
    y_test_wheat, y_wheat_pred
)
print(report_wheat)
```

|               | precision | recall | f1-score | support | 
| -- | -- | -- | -- | -- |
|           0  | 0.82 | 0.82 | 0.82 | 11 |
|           1  | 1.00 | 0.93 | 0.96 | 14 |
|            2 | 0.89 | 0.94 | 0.91 | 17 |
|   accuracy   |      |      | 0.90 | 42 |
|    macro avg | 0.90 | 0.90 | 0.90 | 42 |
| weighted avg | 0.91 | 0.90 | 0.91 | 42 |

```python
conf_mtx_wheat = confusion_matrix(y_test_wheat, y_wheat_pred)
conf_mtx_wheat

# array([[ 9,  0,  2],
#        [ 1, 13,  0],
#        [ 1,  0, 16]])
```

```python
conf_mtx_wheat_plot = ConfusionMatrixDisplay(
    confusion_matrix=conf_mtx_wheat
)

conf_mtx_wheat_plot.plot()
plt.show()
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_07.webp)


### Margin Plots for Support Vector Classifier

```python
# get dataset
!wget https://github.com/alpeshraj/mouse_viral_study/raw/main/mouse_viral_study.csv -P datasets
```

```python
mice_df = pd.read_csv('datasets/mouse_viral_study.csv')
mice_df.head(5)
```

|   | Med_1_mL | Med_2_mL | Virus Present |
| -- | -- | -- | -- |
| 0 | 6.508231 | 8.582531 | 0 |
| 1 | 4.126116 | 3.073459 | 1 |
| 2 | 6.427870 | 6.369758 | 0 |
| 3 | 3.672953 | 4.905215 | 1 |
| 4 | 1.580321 | 2.440562 | 1 |

```python
sns.scatterplot(data=mice_df, x='Med_1_mL',y='Med_2_mL',hue='Virus Present', palette='winter')
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_12.webp)

```python
# visualizing a hyperplane to separate the two features
sns.scatterplot(data=mice_df, x='Med_1_mL',y='Med_2_mL',hue='Virus Present', palette='winter')

x = np.linspace(0,10,100)
m = -1
b = 11
y = m*x + b

plt.plot(x,y,c='fuchsia')
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_13.webp)


#### SVC with a Linear Kernel

```python
# using a support vector classifier to calculate maximize the margin between both classes

y_vir = mice_df['Virus Present']
X_vir = mice_df.drop('Virus Present',axis=1)

# kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
# the smaller the C value the more feature vectors will be inside the margin
model_vir = svm.SVC(kernel='linear', C=1000)

model_vir.fit(X_vir, y_vir)
```

```python
# import helper function
from helper.svm_margin_plot import plot_svm_boundary
```

```python
plot_svm_boundary(model_vir, X_vir, y_vir)
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_14.webp)

```python
# the smaller the C value the more feature vectors will be inside the margin
model_vir_low_reg = svm.SVC(kernel='linear', C=0.005)
model_vir_low_reg.fit(X_vir, y_vir)
plot_svm_boundary(model_vir_low_reg, X_vir, y_vir)
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_15.webp)


#### SVC with a Radial Basis Function Kernel

```python
model_vir_rbf = svm.SVC(kernel='rbf', C=1)
model_vir_rbf.fit(X_vir, y_vir)
plot_svm_boundary(model_vir_rbf, X_vir, y_vir)
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_16.webp)

```python
# # gamma : {'scale', 'auto'} or float, default='scale'
# - if ``gamma='scale'`` (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
# - if 'auto', uses 1 / n_features
# - if float, must be non-negative.
model_vir_rbf_auto_gamma = svm.SVC(kernel='rbf', C=1, gamma='auto')
model_vir_rbf_auto_gamma.fit(X_vir, y_vir)
plot_svm_boundary(model_vir_rbf_auto_gamma, X_vir, y_vir)
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_17.webp)


#### SVC with a Sigmoid Kernel

```python
model_vir_sigmoid = svm.SVC(kernel='sigmoid', gamma='scale')
model_vir_sigmoid.fit(X_vir, y_vir)
plot_svm_boundary(model_vir_sigmoid, X_vir, y_vir)
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_18.webp)


#### SVC with a Polynomial Kernel

```python
model_vir_poly = svm.SVC(kernel='poly', C=1, degree=2)
model_vir_poly.fit(X_vir, y_vir)
plot_svm_boundary(model_vir_poly, X_vir, y_vir)
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_19.webp)


### Grid Search for Support Vector Classifier

```python
svm_base_model = svm.SVC()

param_grid = {
    'C':[0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}
```

```python
grid = GridSearchCV(svm_base_model, param_grid) 
grid.fit(X_vir, y_vir)
```

```python
grid.best_params_
# {'C': 0.01, 'kernel': 'linear'}
```

### Support Vector Regression

```python
# dataset
!wget https://github.com/fsdhakan/ML/raw/main/cement_slump.csv -P datasets
```

```python
cement_df = pd.read_csv('datasets/cement_slump.csv')
cement_df.head(5)
```

|    | Cement | Slag | Fly ash | Water | SP | Coarse Aggr. | Fine Aggr. | SLUMP(cm) | FLOW(cm) | Compressive Strength (28-day)(Mpa) |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 0 | 273.0 | 82.0 | 105.0 | 210.0 | 9.0 | 904.0 | 680.0 | 23.0 | 62.0 | 34.99 |
| 1 | 163.0 | 149.0 | 191.0 | 180.0 | 12.0 | 843.0 | 746.0 | 0.0 | 20.0 | 41.14 |
| 2 | 162.0 | 148.0 | 191.0 | 179.0 | 16.0 | 840.0 | 743.0 | 1.0 | 20.0 | 41.81 |
| 3 | 162.0 | 148.0 | 190.0 | 179.0 | 19.0 | 838.0 | 741.0 | 3.0 | 21.5 | 42.08 |
| 4 | 154.0 | 112.0 | 144.0 | 220.0 | 10.0 | 923.0 | 658.0 | 20.0 | 64.0 | 26.82 |

```python
plt.figure(figsize=(8,8))
sns.heatmap(cement_df.corr(), annot=True, cmap='viridis')
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_20.webp)

```python
# drop labels
X_cement = cement_df.drop('Compressive Strength (28-day)(Mpa)', axis=1)
y_cement = cement_df['Compressive Strength (28-day)(Mpa)']
```

```python
 # train/test split
    X_train_cement, X_test_cement, y_train_cement, y_test_cement = train_test_split(
     X_cement,
     y_cement,
     test_size=0.3,
     random_state=42
 )
```

```python
# normalize
scaler = StandardScaler()
X_train_cement_scaled = scaler.fit_transform(X_train_cement)
X_test_cement_scaled = scaler.transform(X_test_cement)
```

#### Base Model Run

```python
base_model_cement = svm.SVR()
```

```python
base_model_cement.fit(X_train_cement_scaled, y_train_cement)

base_model_predictions = base_model_cement.predict(X_test_cement_scaled)
```

```python
mae = mean_absolute_error(y_test_cement, base_model_predictions)
rmse = mean_squared_error(y_test_cement, base_model_predictions)
mean_abs = y_test_cement.mean()
avg_error = mae * 100 / mean_abs

print('MAE: ', mae.round(2), 'RMSE: ', rmse.round(2), 'Relative Avg. Error: ', avg_error.round(2), '%')
```

| MAE | RMSE |  Relative Avg. Error |
| -- | -- | -- |
| 4.68 | 36.95 | 12.75 % |


#### Grid Search for better Hyperparameter

```python
param_grid = {
    'C': [0.001,0.01,0.1,0.5,1],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2,3,4],
    'epsilon': [0,0.01,0.1,0.5,1,2]
}
```

```python
cement_grid = GridSearchCV(base_model_cement, param_grid)
cement_grid.fit(X_train_cement_scaled, y_train_cement)
```

```python
cement_grid.best_params_
# {'C': 1, 'degree': 2, 'epsilon': 2, 'gamma': 'scale', 'kernel': 'linear'}
```

```python
cement_grid_predictions = cement_grid.predict(X_test_cement_scaled)
```

```python
mae_grid = mean_absolute_error(y_test_cement, cement_grid_predictions)
rmse_grid = mean_squared_error(y_test_cement, cement_grid_predictions)
mean_abs = y_test_cement.mean()
avg_error_grid = mae_grid * 100 / mean_abs

print('MAE: ', mae_grid.round(2), 'RMSE: ', rmse_grid.round(2), 'Relative Avg. Error: ', avg_error_grid.round(2), '%')
```

| MAE | RMSE |  Relative Avg. Error |
| -- | -- | -- |
| 1.85 | 5.2 | 5.05 % |


### Example Task - Wine Fraud

#### Data Exploration

```python
# dataset
!wget https://github.com/CAPGAGA/Fraud-in-Wine/raw/main/wine_fraud.csv -P datasets
```

```python
wine_df = pd.read_csv('datasets/wine_fraud.csv')
wine_df.head(5)
```

|  | fixed acidity | volatile acidity | citric acid | residual sugar | chlorides | free sulfur dioxide | total sulfur dioxide | density | pH | sulphates | alcohol | quality | type |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 0 | 7.4 | 0.70 | 0.00 | 1.9 | 0.076 | 11.0 | 34.0 | 0.9978 | 3.51 | 0.56 | 9.4 | Legit | red |
| 1 | 7.8 | 0.88 | 0.00 | 2.6 | 0.098 | 25.0 | 67.0 | 0.9968 | 3.20 | 0.68 | 9.8 | Legit | red |
| 2 | 7.8 | 0.76 | 0.04 | 2.3 | 0.092 | 15.0 | 54.0 | 0.9970 | 3.26 | 0.65 | 9.8 | Legit | red |
| 3 | 11.2 | 0.28 | 0.56 | 1.9 | 0.075 | 17.0 | 60.0 | 0.9980 | 3.16 | 0.58 | 9.8 | Legit | red |
| 4 | 7.4 | 0.70 | 0.00 | 1.9 | 0.076 | 11.0 | 34.0 | 0.9978 | 3.51 | 0.56 | 9.4 | Legit | red |

```python
wine_df.value_counts('quality')
```

| quality | |
| -- | -- |
| Legit | 6251 |
| Fraud  | 246 |
_dtype: int64_

```python
wine_df['quality'].value_counts().plot(
    kind='bar',
    figsize=(10,5), 
    title='Wine - Quality distribution')

```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_21.webp)

```python
plt.figure(figsize=(10, 5))
plt.title('Wine - Quality distribution by Type')

sns.countplot(
    data=wine_df,
    x='quality',
    hue='type',
    palette='winter'
)

plt.savefig('assets/Scikit_Learn_22.webp', bbox_inches='tight')
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_22.webp)

```python
wine_df_white = wine_df[wine_df['type'] == 'white']
wine_df_red = wine_df[wine_df['type'] == 'red']
```

```python
# fraud percentage by wine type
legit_white_wines = wine_df_white.value_counts('quality')[0]
fraud_white_wines = wine_df_white.value_counts('quality')[1]
white_fraud_percentage = fraud_white_wines * 100 / (legit_white_wines + fraud_white_wines)

legit_red_wines = wine_df_red.value_counts('quality')[0]
fraud_red_wines = wine_df_red.value_counts('quality')[1]
red_fraud_percentage = fraud_red_wines * 100 / (legit_red_wines + fraud_red_wines)

print(
    'Fraud Percentage: \nWhite Wines: ',
    white_fraud_percentage.round(2),
    '% \nRed Wines: ',
    red_fraud_percentage.round(2),
    '%'
)
```

| Fraud Percentage: | |
| -- | -- |
| White Wines: | 3.74 % |
| Red Wines: | 3.94 % |

```python
# make features numeric
feature_map = {
    'Legit': 0,
    'Fraud': 1,
    'red': 0,
    'white': 1
}

wine_df['quality_enc'] = wine_df['quality'].map(feature_map)
wine_df['type_enc'] = wine_df['type'].map(feature_map)
wine_df[['quality', 'quality_enc', 'type', 'type_enc']]
```

|  | quality | quality_enc | type | type_enc |
| -- | -- | -- | -- | -- |
| 0 | Legit | 0 | red | 0 |
| 1 | Legit | 0 | red | 0 |
| 2 | Legit | 0 | red | 0 |
| 3 | Legit | 0 | red | 0 |
| 4 | Legit | 0 | red | 0 |
| ... |
| 6492 | Legit | 0 | white | 1 |
| 6493 | Legit | 0 | white | 1 |
| 6494 | Legit | 0 | white | 1 |
| 6495 | Legit | 0 | white | 1 |
| 6496 | Legit | 0 | white | 1 |
_6497 rows × 4 columns_

```python
# find correlations
wine_df.corr(numeric_only=True)
```

|  | fixed acidity | volatile acidity | citric acid | residual sugar | chlorides | free sulfur dioxide | total sulfur dioxide | density | pH | sulphates | alcohol | quality_enc | type_enc |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| fixed acidity | 1.000000 | 0.219008 | 0.324436 | -0.111981 | 0.298195 | -0.282735 | -0.329054 | 0.458910 | -0.252700 | 0.299568 | -0.095452 | 0.021794 | -0.486740 |
| volatile acidity | 0.219008 | 1.000000 | -0.377981 | -0.196011 | 0.377124 | -0.352557 | -0.414476 | 0.271296 | 0.261454 | 0.225984 | -0.037640 | 0.151228 | -0.653036 |
| citric acid | 0.324436 | -0.377981 | 1.000000 | 0.142451 | 0.038998 | 0.133126 | 0.195242 | 0.096154 | -0.329808 | 0.056197 | -0.010493 | -0.061789 | 0.187397 |
| residual sugar | -0.111981 | -0.196011 | 0.142451 | 1.000000 | -0.128940 | 0.402871 | 0.495482 | 0.552517 | -0.267320 | -0.185927 | -0.359415 | -0.048756 | 0.348821 |
| chlorides | 0.298195 | 0.377124 | 0.038998 | -0.128940 | 1.000000 | -0.195045 | -0.279630 | 0.362615 | 0.044708 | 0.395593 | -0.256916 | 0.034499 | -0.512678 |
| free sulfur dioxide | -0.282735 | -0.352557 | 0.133126 | 0.402871 | -0.195045 | 1.000000 | 0.720934 | 0.025717 | -0.145854 | -0.188457 | -0.179838 | -0.085204 | 0.471644 |
| total sulfur dioxide | -0.329054 | -0.414476 | 0.195242 | 0.495482 | -0.279630 | 0.720934 | 1.000000 | 0.032395 | -0.238413 | -0.275727 | -0.265740 | -0.035252 | 0.700357 |
| density | 0.458910 | 0.271296 | 0.096154 | 0.552517 | 0.362615 | 0.025717 | 0.032395 | 1.000000 | 0.011686 | 0.259478 | -0.686745 | 0.016351 | -0.390645 |
| pH | -0.252700 | 0.261454 | -0.329808 | -0.267320 | 0.044708 | -0.145854 | -0.238413 | 0.011686 | 1.000000 | 0.192123 | 0.121248 | 0.020107 | -0.329129 |
| sulphates | 0.299568 | 0.225984 | 0.056197 | -0.185927 | 0.395593 | -0.188457 | -0.275727 | 0.259478 | 0.192123 | 1.000000 | -0.003029 | -0.034046 | -0.487218 |
| alcohol | -0.095452 | -0.037640 | -0.010493 | -0.359415 | -0.256916 | -0.179838 | -0.265740 | -0.686745 | 0.121248 | -0.003029 | 1.000000 | -0.051141 | 0.032970 |
| quality_enc | 0.021794 | 0.151228 | -0.061789 | -0.048756 | 0.034499 | -0.085204 | -0.035252 | 0.016351 | 0.020107 | -0.034046 | -0.051141 | 1.000000 | -0.004598 |
| type_enc | -0.486740 | -0.653036 | 0.187397 | 0.348821 | -0.512678 | 0.471644 | 0.700357 | -0.390645 | -0.329129 | -0.487218 | 0.032970 | -0.004598 | 1.000000 |

```python
plt.figure(figsize=(12,8))
sns.heatmap(wine_df.corr(numeric_only=True), annot=True, cmap='viridis')
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_23.webp)

```python
# how does the quality correlate to measurements
wine_df.corr(numeric_only=True)['quality_enc']
```

| Quality Correlstion | |
| -- | -- |
| fixed acidity        |  0.021794 |
| volatile acidity     |  0.151228 |
| citric acid          | -0.061789 |
| residual sugar       | -0.048756 |
| chlorides            |  0.034499 |
| free sulfur dioxide  | -0.085204 |
| total sulfur dioxide | -0.035252 |
| density              |  0.016351 |
| pH                   |  0.020107 |
| sulphates            | -0.034046 |
| alcohol              | -0.051141 |
| quality_enc          |  1.000000 |
| type_enc             | -0.004598 |
_Name: quality_enc, dtype: float64_

```python
wine_df.corr(numeric_only=True)['quality_enc'][:-2].sort_values().plot(
    figsize=(12,5),
    kind='bar',
    title='Correlation of Measurements to Quality'
)
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_24.webp)


#### Regression Model

```python
# separate target + remove string values
X_wine = wine_df.drop(['quality_enc', 'quality', 'type'], axis=1)
y_wine = wine_df['quality']

print(X_wine.shape, y_wine.shape)
```

```python
# train-test split
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(
    X_wine,
    y_wine,
    test_size=0.1,
    random_state=42
)
```

```python
# normalization
scaler = StandardScaler()
X_wine_train_scaled = scaler.fit_transform(X_wine_train)
X_wine_test_scaled = scaler.transform(X_wine_test)
```

```python
# create the SVC model using class_weight to balance out the
# dataset that heavily leaning towards non-frauds
svc_wine_base = svm.SVC(
    kernel='rbf',
    class_weight='balanced'
)
```

```python
# grid search
param_grid = {
    'C': [0.5, 1, 1.5, 2, 2.5],
    'gamma' : ['scale', 'auto']
}

wine_grid = GridSearchCV(svc_wine_base, param_grid)
wine_grid.fit(X_wine_train_scaled, y_wine_train)
print('Best Params: ', wine_grid.best_params_)
# Best Params:  {'C': 2.5, 'gamma': 'auto'}
```

```python
y_wine_pred = wine_grid.predict(X_wine_test_scaled)
```

```python
print(
    'Accuracy Score: ',
    accuracy_score(y_wine_test, y_wine_pred, normalize=True).round(4)*100, '%'
)
# Accuracy Score:  84.77 %
```

```python
report_wine = classification_report(
    y_wine_test, y_wine_pred
)
print(report_wine)
```

|         | precision | recall | f1-score | support |
| --      | -- | -- | -- | -- |
| Fraud   | 0.16 | 0.68 | 0.26 |  25 |
| Legit   | 0.99 | 0.85 | 0.92 | 625 |
|     accuracy |      |      | 0.85 | 650 |
|    macro avg | 0.57 | 0.77 | 0.59 | 650 |
| weighted avg | 0.95 | 0.85 | 0.89 | 650 |

```python
conf_mtx_wine = confusion_matrix(y_wine_test, y_wine_pred)
conf_mtx_wine

# array([[ 17,   8],
#        [ 91, 534]])
```

```python
conf_mtx_wine_plot = ConfusionMatrixDisplay(
    confusion_matrix=conf_mtx_wine
)

conf_mtx_wine_plot.plot(cmap='plasma')
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_25.webp)

```python
# expand grid search
param_grid = {
    'C': [1000, 1050, 1100, 1050, 1200],
    'gamma' : ['scale', 'auto']
}

wine_grid = GridSearchCV(svc_wine_base, param_grid)
wine_grid.fit(X_wine_train_scaled, y_wine_train)
print('Best Params: ', wine_grid.best_params_)
# Best Params:  {'C': 1100, 'gamma': 'scale'}
```

```python
y_wine_pred = wine_grid.predict(X_wine_test_scaled)
print('Accuracy Score: ',accuracy_score(y_wine_test, y_wine_pred, normalize=True).round(4)*100, '%')
# Accuracy Score:  94.31 %
report_wine = classification_report(y_wine_test, y_wine_pred)
print(report_wine)
conf_mtx_wine = confusion_matrix(y_wine_test, y_wine_pred)

conf_mtx_wine_plot = ConfusionMatrixDisplay(
    confusion_matrix=conf_mtx_wine
)

conf_mtx_wine_plot.plot(cmap='plasma')
```

<!-- #region -->
|         | precision | recall | f1-score | support |
| --      | -- | -- | -- | -- |
| Fraud   | 0.29 | 0.32 | 0.30 |  25 |
| Legit   | 0.97 | 0.97 | 0.97 | 625 |
|     accuracy |      |      | 0.85 | 650 |
|    macro avg | 0.63 | 0.64 | 0.64 | 650 |
| weighted avg | 0.95 | 0.94 | 0.94 | 650 |


![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_26.webp)
<!-- #endregion -->

## Boosting Methods

```python
# dataset - label mushrooms as poisonous or eatable
!wget https://github.com/semnan-university-ai/Mushroom/raw/main/Mushroom.csv -P datasets
```

### Dataset Exploration

```python
shroom_df = pd.read_csv('datasets/mushrooms.csv')
shroom_df.head(5).transpose()
```

<!-- #region -->
[Mushroom Data Set](https://archive.ics.uci.edu/ml/datasets/mushroom)

1. __cap-shape__: bell = `b`, conical = `c`, convex = `x`, flat = `f`,  knobbed = `k`, sunken = `s`
2. __cap-surface__: fibrous = `f`, grooves = `g`, scaly = `y`, smooth = `s`
3. __cap-color__: brown = `n`, buff = `b`, cinnamon = `c`, gray = `g`, green = `r`,  pink = `p`, purple = `u`, red = `e`, white = `w`, yellow = `y`
4. __bruises?__: bruises = `t`, no = `f`
5. __odor__: almond = `a`, anise = `l`, creosote = `c`, fishy = `y`, foul = `f`,  musty = `m`, none = `n`, pungent = `p`, spicy = `s`
6. __gill-attachment__: attached = `a`, descending = `d`, free = `f`, notched = `n`
7. __gill-spacing__: close = `c`, crowded = `w`, distant = `d`
8. __gill-size__: broad = `b`, narrow = `n`
9. __gill-color__: black = `k`, brown = `n`, buff = `b`, chocolate = `h`, gray = `g`,  green = `r`, orange = `o`, pink = `p`, purple = `u`, red = `e`,  white = `w`, yellow = `y`
10. __stalk-shape__: enlarging = `e`, tapering = `t`
11. __stalk-root__: bulbous = `b`, club = `c`, cup = `u`, equal = `e`,  rhizomorphs = `z`, rooted = `r`, missing = `?`
12. __stalk-surface-above-ring__: fibrous = `f`, scaly = `y`, silky = `k`, smooth = `s`
13. __stalk-surface-below-ring__: fibrous = `f`, scaly = `y`, silky = `k`, smooth = `s`
14. __stalk-color-above-ring__: brown = `n`, buff = `b`, cinnamon = `c`, gray = `g`, orange = `o`,  pink = `p`, red = `e`, white = `w`, yellow = `y`
15. __stalk-color-below-ring__: brown = `n`, buff = `b`, cinnamon = `c`, gray = `g`, orange = `o`,  pink = `p`, red = `e`, white = `w`, yellow = `y`
16. __veil-type__: partial = `p`, universal = `u`
17. __veil-color__: brown = `n`, orange = `o`, white = `w`, yellow = `y`
18. __ring-number__: none = `n`, one = `o`, two = `t`
19. __ring-type__: cobwebby = `c`, evanescent = `e`, flaring = `f`, large = `l`,  none = `n`, pendant = `p`, sheathing = `s`, zone = `z`
20. __spore-print-color__: black = `k`, brown = `n`, buff = `b`, chocolate = `h`, green = `r`,  orange = `o`, purple = `u`, white = `w`, yellow = `y`
21. __population__: abundant = `a`, clustered = `c`, numerous = `n`,  scattered = `s`, several = `v`, solitary = `y
22. __habitat__: grasses = `g`, leaves = `l`, meadows = `m`, paths = `p`,  urban = `u`, waste = `w`, woods = `d`


|  | 0 | 1 | 2 | 3 | 4 |
| -- | -- | -- | -- | -- | -- |
| class | p | e | e | p | e |
| cap-shape | x | x | b | x | x |
| cap-surface | s | s | s | y | s |
| cap-color | n | y | w | w | g |
| bruises | t | t | t | t | f |
| odor | p | a | l | p | n |
| gill-attachment | f | f | f | f | f |
| gill-spacing | c | c | c | c | w |
| gill-size | n | b | b | n | b |
| gill-color | k | k | n | n | k |
| stalk-shape | e | e | e | e | t |
| stalk-root | e | c | c | e | e |
| stalk-surface-above-ring | s | s | s | s | s |
| stalk-surface-below-ring | s | s | s | s | s |
| stalk-color-above-ring | w | w | w | w | w |
| stalk-color-below-ring | w | w | w | w | w |
| veil-type | p | p | p | p | p |
| veil-color | w | w | w | w | w |
| ring-number | o | o | o | o | o |
| ring-type | p | p | p | p | e |
| spore-print-color | k | n | n | k | n |
| population | s | n | n | s | a |
| habitat | u | g | m | u | g |
<!-- #endregion -->

```python
shroom_df.isnull().sum()
```

| | |
| -- | -- |
| class | 0 |
| cap-shape | 0 |
| cap-surface | 0 |
| cap-color | 0 |
| bruises | 0 |
| odor | 0 |
| gill-attachment | 0 |
| gill-spacing | 0 |
| gill-size | 0 |
| gill-color | 0 |
| stalk-shape | 0 |
| stalk-root | 0 |
| stalk-surface-above-ring | 0 |
| stalk-surface-below-ring | 0 |
| stalk-color-above-ring | 0 |
| stalk-color-below-ring | 0 |
| veil-type | 0 |
| veil-color | 0 |
| ring-number | 0 |
| ring-type | 0 |
| spore-print-color | 0 |
| population | 0 |
| habitat | 0 |
_dtype: int64_

```python
feature_df = shroom_df.describe().transpose().reset_index(
    names=['feature']
).sort_values(
    'unique', ascending=False
)
```

|    | feature | count | unique | top | freq |
| -- | -- | -- | -- | -- | -- |
| 9 | gill-color | 8124 | 12 | b | 1728 |
| 3 | cap-color | 8124 | 10 | n | 2284 |
| 20 | spore-print-color | 8124 | 9 | w | 2388 |
| 5 | odor | 8124 | 9 | n | 3528 |
| 15 | stalk-color-below-ring | 8124 | 9 | w | 4384 |
| 14 | stalk-color-above-ring | 8124 | 9 | w | 4464 |
| 22 | habitat | 8124 | 7 | d | 3148 |
| 1 | cap-shape | 8124 | 6 | x | 3656 |
| 21 | population | 8124 | 6 | v | 4040 |
| 19 | ring-type | 8124 | 5 | p | 3968 |
| 11 | stalk-root | 8124 | 5 | b | 3776 |
| 12 | stalk-surface-above-ring | 8124 | 4 | s | 5176 |
| 13 | stalk-surface-below-ring | 8124 | 4 | s | 4936 |
| 17 | veil-color | 8124 | 4 | w | 7924 |
| 2 | cap-surface | 8124 | 4 | y | 3244 |
| 18 | ring-number | 8124 | 3 | o | 7488 |
| 10 | stalk-shape | 8124 | 2 | t | 4608 |
| 8 | gill-size | 8124 | 2 | b | 5612 |
| 7 | gill-spacing | 8124 | 2 | c | 6812 |
| 6 | gill-attachment | 8124 | 2 | f | 7914 |
| 4 | bruises | 8124 | 2 | f | 4748 |
| 0 | class | 8124 | 2 | e | 4208 |
| 16 | veil-type | 8124 | 1 | p | 8124 |

```python
plt.figure(figsize=(12,8))
plt.title('Mushroom Features :: Number of unique Features')
sns.barplot(data=feature_df, y='feature', x='unique', orient='h', palette='summer_r')
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_47.webp)

```python
plt.figure(figsize=(10,4))
plt.title('Mushroom Count :: Editable vs Poisonous')
sns.countplot(data=shroom_df, x='class', palette='seismic_r')
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_46.webp)


### Adaptive Boosting

```python
# remove lable class
X_shroom = shroom_df.drop('class', axis=1)
# make all values numeric
X_shroom = pd.get_dummies(X_shroom, drop_first=True)

y_shroom = shroom_df['class']
```

```python
# train/test split
X_shroom_train, X_shroom_test, y_shroom_train, y_shroom_test = train_test_split(
    X_shroom,
    y_shroom,
    test_size=0.15,
    random_state=42
)
```

#### Feature Exploration

```python
# don't try fit a perfect model but only return
# the most important feature for classification
abc_shroom = AdaBoostClassifier(estimator=None, n_estimators=1)
abc_shroom.fit(X_shroom_train,y_shroom_train)
```

```python
shroom_preds = abc_shroom.predict(X_shroom_test)

print('Accuracy Score: ',accuracy_score(y_shroom_test, shroom_preds, normalize=True).round(4)*100, '%')
# Accuracy Score:  88.35 %
```

```python
report_shroom = classification_report(y_shroom_test, shroom_preds)
print(report_shroom)
```

|  | precision | recall | f1-score | support |
| -- | -- | -- | -- | -- |
|     e | 0.97 | 0.80 | 0.88 | 637 |
|     p | 0.82 | 0.97 | 0.89 | 582 |
|    accuracy |  |  | 0.88 | 1219 |
|   macro avg | 0.89 | 0.89 | 0.88 | 1219 |
|weighted avg | 0.90 | 0.88 | 0.88 | 1219 |

```python
conf_mtx_shroom = confusion_matrix(y_shroom_test, shroom_preds)

conf_mtx_shroom_plot = ConfusionMatrixDisplay(
    confusion_matrix=conf_mtx_shroom
)

conf_mtx_shroom_plot.plot(cmap='winter_r')
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_48.webp)

```python
# the model was fit on a single feature and still resulted in a pretty good performance.
# Let's find out what feature was chosen for the classification.

shroom_index = ['importance']
shroom_data_columns = pd.Series(X_shroom.columns)
shroom_importance_array = abc_shroom.feature_importances_
shroom_importance_df = pd.DataFrame(shroom_importance_array, shroom_data_columns, shroom_index)
shroom_importance_df.value_counts()
```

| importance | count |
| -- | -- |
| 0.0 | 94 |
| 1.0 | 1 |
_dtype: int64_

```python
# plot a slice of the dataframe to find the feature
shroom_importance_df_sorted = shroom_importance_df.sort_values(
    by='importance',
    ascending=True
)

shroom_importance_df_sorted[-5:].plot(
    kind='barh',
    title='Feature Importance for Mushroom Classification',
    figsize=(8,4)
)
```

The most important feature (as determined by the model) is the odor - in this case a odor of `none` is the best indicator to classify a poisonous mushroom:

> odor: almond = a, anise = l, creosote = c, fishy = y, foul = f, musty = m, none = n, pungent = p, spicy = s

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_49.webp)

```python
# the mojority of poisonous mushrooms do have an odor
# naking the lack of it a good indicator for an eatable variety
plt.figure(figsize=(12,4))
plt.title('Mushroom Odor vs Class')
sns.countplot(data=shroom_df, x='odor', hue='class', palette='summer')
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_50.webp)


#### Optimizing Hyperparameters

```python
# find out how many of the 95 features you have
# to add to your model to get a better fit

error_rates = []

for estimators in range(1,96):
    model = AdaBoostClassifier(n_estimators=estimators)
    model.fit(X_shroom_train,y_shroom_train)
    preds = model.predict(X_shroom_test)
    
    err = 1 - accuracy_score(y_shroom_test, preds)
    error_rates.append(err)
```

```python
x_range=range(1,96)
plt.figure(figsize=(10,4))
plt.title('Adaboost Error Rate vs n_estimators')
plt.xlabel('n_estimators')
plt.ylabel('Error Rate')
plt.xticks(np.arange(min(x_range), max(x_range)+1, 3.0))
plt.plot(x_range, error_rates)
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_51.webp)

```python
# already after 16 estimators there is no
# visible improvment for the error rate
abc_shroom2 = AdaBoostClassifier(estimator=None, n_estimators=16)
abc_shroom2.fit(X_shroom_train,y_shroom_train)

shroom_preds2 = abc_shroom2.predict(X_shroom_test)

print('Accuracy Score: ',accuracy_score(y_shroom_test, shroom_preds2, normalize=True).round(4)*100, '%')
# Accuracy Score:  99.92 %

report_shroom2 = classification_report(y_shroom_test, shroom_preds2)
print(report_shroom2)
```

|  | precision | recall | f1-score | support |
| -- | -- | -- | -- | -- |
|     e | 1.00 | 1.00 | 1.00 | 637 |
|     p | 1.00 | 1.00 | 1.00 | 582 |
|    accuracy |  |  | 1.00 | 1219 |
|   macro avg | 1.00 | 1.00 | 1.00 | 1219 |
|weighted avg | 1.00 | 1.00 | 1.00 | 1219 |

```python
conf_mtx_shroom2 = confusion_matrix(y_shroom_test, shroom_preds2)

conf_mtx_shroom_plot2 = ConfusionMatrixDisplay(
    confusion_matrix=conf_mtx_shroom2
)

conf_mtx_shroom_plot2.plot(cmap='winter_r')
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_52.webp)

```python
shroom_index = ['importance']
shroom_data_columns = pd.Series(X_shroom.columns)
shroom_importance_array = abc_shroom2.feature_importances_
shroom_importance_df = pd.DataFrame(shroom_importance_array, shroom_data_columns, shroom_index)
shroom_importance_df.value_counts()

# there are 12 features now that are deemed important
```

| importance | count |
| -- | -- |
| 0.0000 | 83 |
| 0.0625 |  9 |
| 0.1250  |  2 |
| 0.1875  |  1 |
_dtype: int64_

```python
shroom_importance_df_sorted = shroom_importance_df.sort_values(
    by='importance',
    ascending=True
).tail(13)
```

|  | importance |
| -- | -- |
| gill-size_n | 0.1875 |
| population_v | 0.1250 |
| odor_n | 0.1250 |
| odor_c | 0.0625 |
| stalk-shape_t | 0.0625 |
| spore-print-color_w | 0.0625 |
| population_c | 0.0625 |
| ring-type_p | 0.0625 |
| spore-print-color_r | 0.0625 |
| stalk-surface-above-ring_k | 0.0625 |
| gill-spacing_w | 0.0625 |
| odor_f | 0.0625 |
| stalk-color-below-ring_w | 0.0000 |

```python
plt.figure(figsize=(10,6))
plt.title('Features important to classify poisonous Mushrooms')

sns.barplot(
    data=shroom_importance_df_sorted.tail(13),
    y=shroom_importance_df_sorted.tail(13).index,
    x='importance',
    orient='h',
    palette='summer'
)
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_53.webp)


### Gradient Boosting

####  Gridsearch for best Hyperparameter

```python
gb_shroom = GradientBoostingClassifier()
```

```python
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05,0.1,0.2],
    'max_depth': [2,3,4,5]
}
```

```python
shroom_grid = GridSearchCV(gb_shroom, param_grid)
shroom_grid.fit(X_shroom_train, y_shroom_train)
shroom_grid.best_params_
# {'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 150}
```

```python
shroom_grid_preds = shroom_grid.predict(X_shroom_test)

print('Accuracy Score: ',accuracy_score(y_shroom_test, shroom_grid_preds, normalize=True).round(4)*100, '%')
# Accuracy Score:  100.0 %

report_shroom_grid_preds = classification_report(y_shroom_test, shroom_grid_preds)
print(report_shroom_grid_preds)
```

|  | precision | recall | f1-score | support |
| -- | -- | -- | -- | -- |
|     e | 1.00 | 1.00 | 1.00 | 637 |
|     p | 1.00 | 1.00 | 1.00 | 582 |
|    accuracy |  |  | 1.00 | 1219 |
|   macro avg | 1.00 | 1.00 | 1.00 | 1219 |
|weighted avg | 1.00 | 1.00 | 1.00 | 1219 |

```python
conf_mtx_shroom_grid = confusion_matrix(y_shroom_test, shroom_grid_preds)

conf_mtx_shroom_grid_plot = ConfusionMatrixDisplay(
    confusion_matrix=conf_mtx_shroom_grid
)

conf_mtx_shroom_grid_plot.plot(cmap='winter_r')
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_54.webp)


#### Feature Importance

```python
shroom_feature_importance = shroom_grid.best_estimator_.feature_importances_
feature_importance_df = pd.DataFrame(
    index = X_shroom.columns,
    data = shroom_feature_importance,
    columns = ['importance']
)

# kick all features that have zero importance and sort by importance
feature_importance_df = feature_importance_df[
    feature_importance_df['importance'] > 3e-03
].sort_values(
    by='importance',
    ascending=False
)
```

```python
plt.figure(figsize=(10,6))
plt.title('Features important to classify poisonous Mushrooms')

sns.barplot(
    data=feature_importance_df,
    y=feature_importance_df.index,
    x='importance',
    orient='h',
    palette='summer'
)
```

![scikit-learn - Machine Learning in Python](https://github.com/mpolinowski/python-scikitlearn-cheatsheet/raw/master/assets/Scikit_Learn_55.webp)



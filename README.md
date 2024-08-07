# Machine Learning 

## Installation
Used all the packages mentioned in requirements.txt on this repo

# Supervised Learning
y=mx+b
Predict Y based on x which is a labelled data.

## Train and Split
Split Train and Test data from given set of features

```bash
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

## Linear Regression

```bash
from sklearn.linear_model import LinearRegression
```

### Metrics
```bash
from sklearn.metrics import mean_absolute_error,mean_squared_error
```

MAE is the easiest to understand, because it's the average error.
MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.

## Polynomial Regression
```bash
from sklearn.preprocessing import PolynomialFeatures
polynomial_converter = PolynomialFeatures(degree=2,include_bias=False)
poly_features = polynomial_converter.fit_transform(X)
```

## Scaling the Data
While our particular data set has all the values in the same order of magnitude ($1000s of dollars spent)
typically that won't be the case on a dataset, and since the mathematics behind regularized models will sum coefficients 
together, its important to standardize the features. 

```bash
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

```bash
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=10)
```

```bash
from sklearn.linear_model import RidgeCV
ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10.0),scoring='neg_mean_absolute_error')
```

```bash
from sklearn.linear_model import LassoCV
lasso_cv_model = LassoCV(eps=0.1,n_alphas=100,cv=5)
```

```bash
from sklearn.linear_model import ElasticNetCV
elastic_model = ElasticNetCV(l1_ratio=[.1, .5, .7,.9, .95, .99, 1],tol=0.01)
```

## KNN 
It works by finding the K nearest points in the training dataset and uses their class to predict the class or value of a new data point.

```bash
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=1)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
```

## Support Vector machines

A support vector machine (SVM) is defined as a machine learning algorithm that uses supervised learning models to solve complex classification, regression, and outlier detection problems by performing optimal data transformations that determine boundaries between data points based on predefined classes, labels, or outputs.

```bash
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1000)
```

### Metrics (to draw SVM boundaries)

```bash
from svm_margin_plot import plot_svm_boundary
plot_svm_boundary(model,X,y)
```

## Decision Tree
A decision tree is a supervised learning algorithm that is used for classification and regression modeling. Regression is a method used for predictive modeling, so these trees are used to either classify data or predict what will come next.

```bash
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix
```

### Visualize the tree

```bash
from sklearn.tree import plot_tree
```

## Random Forest
In a random forest classification, multiple decision trees are created using different random subsets of the data and features. Each decision tree is like an expert, providing its opinion on how to classify the data.

```bash
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10,max_features='auto',random_state=101)
```

## Boosting 

The boosting algorithm assesses model predictions and increases the weight of samples with a more significant error. 

```bash
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=1)
from sklearn.ensemble import GradientBoostingClassifier
```

### Metrics
```bash
from sklearn.metrics import classification_report,plot_confusion_matrix,accuracy_score
```


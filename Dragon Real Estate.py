# Dragon Real Estate - Price Prediction Analysis

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk

housing_df = pd.read_csv("data.csv")
housing_df.head()
housing_df.info()
housing_df["CHAS"].value_counts()
housing_df.describe()

# Train-Test-Split (Coding the underlying function)

def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_size = int(len(data)* test_ratio)
    test_indices = shuffled[:test_size]
    train_indices = shuffled[test_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing_df,0.2)
print(f"Number of rows in training set: {len(train_set)}\nNumber of rows in test set: {len(test_set)}")

# Train-Test-Split (Using the built-in function)

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing_df, test_size = 0.2, random_state = 42)
print(f"Number of rows in training set: {len(train_set)}\nNumber of rows in test set: {len(test_set)}")

# Stratified sampling to deal with "CHAS" categorical feature

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42 )
for train_index, test_index in split.split(housing_df, housing_df["CHAS"]):
    strat_train_set = housing_df.loc[train_index]
    strat_test_set = housing_df.loc[test_index]
    
print("Training set value count : ")    
print(strat_train_set["CHAS"].value_counts())
print("\n\n")
print("Test set value count : ")  
print(strat_test_set["CHAS"].value_counts())

housing_df = strat_train_set.copy()

# Correlation in the data

corr_matrix = housing_df.corr()
corr_matrix["MEDV"].sort_values(ascending = False)

# Combining attributes

housing_df["TPM"] = housing_df["TAX"]/housing_df["RM"]
corr_matrix = housing_df.corr()
corr_matrix["MEDV"].sort_values(ascending = False)

housing_df = strat_train_set.drop("MEDV", axis = 1)
housing_labels = strat_train_set["MEDV"].copy()

# Missing attributes(the concept)

# Three options
# 1. Remove the missing data points
# 2. Remove whole attribute
# 3. Replace missing data points by some value(0, mean or median)

op1 = housing_df.dropna(subset=["RM"])
op1.shape

op2 = housing_df.drop("RM", axis = 1)
op2.shape

median = housing_df["RM"].median()
op3 = housing_df.fillna(median)
op3.shape

# Missing attributes(the built-in function method)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing_df)

imputer.statistics_

a = imputer.transform(housing_df)
housing_tr = pd.DataFrame(a, columns = housing_df.columns)
housing_tr.info()


# Scikit-learn design

# Three kind of objects

# 1. Estimators : estimates some parameter based on a dataset, eg. SimpleImputer.
# Consists of fit() and transform() methods. Fit method fits the dataset and calculates the internal parameters or hyperparameters.


# 2. Transformers : takes input and returns output based on findings from the fit() method.
# Has a convenience function called fit_transform() which is faster than individual functions calls of fit() and transform().


# 3. Predictors : outputs possible value for the target variable based on model, eg. LinearRegression(), KNearestNeighbors().
# Has a fit() and predict() functions. Also has a score()function that evaluates the accuracy of the predictions.


# Feature Scaling

# Primarily, two types of feature scaling methods

# 1. Normalization (Min-Max Scaling) : value scales down to values between 0 and 1. Sklearn provides MinMaxScaler()
#    (value-min)/(max-min)


# 2. Standardization : variance is 1. Sklearn provides StandardScaler()
#    (value-mean)/(standard deviation) 


# Creating a pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

housing_num_tr = my_pipeline.fit_transform(housing_df)
housing_num_tr

# Selecting the desired model for Dragon Real Estates

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression() 
#model = DecisionTreeRegressor() 
model = RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)

some_data = housing_df.iloc[:5]
some_labels = housing_labels[:5]
prep_data = my_pipeline.transform(some_data)
model.predict(prep_data)
list(some_labels)

# Evaluating the model

from sklearn.metrics import mean_squared_error
housing_pred = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing_pred)
rmse = np.sqrt(mse)
rmse

# Using better evaluation (Cross-validation) to avoid overfitting

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring = "neg_mean_squared_error", cv =10)
rmse_scores =np.sqrt(-scores) 
rmse_scores

def print_scores(scores):
    print("Scores : ", scores)
    print("Mean score : ", scores.mean())
    print("Std of score : ", scores.std())

print_scores(rmse_scores)

# Saving the model

from joblib import dump, load
dump(model,"Dragon.joblib")

# Testing the model on test data

X_test = strat_test_set.drop("MEDV", axis =1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prep = my_pipeline.transform(X_test)
final_pred = model.predict(X_test_prep)
final_mse = mean_squared_error(Y_test,final_pred)
final_rmse = np.sqrt(final_mse)
final_rmse

# Model Usage

import numpy as np
from joblib import dump, load
model = load("Dragon.joblib")

features = np.array([[-0.43942006,  7.12628155, -5.12165014, -0.27288841, -1.42262747,
       -0.23894515, -4.31238772,  2.61111401, -1.0016859 , -2.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)







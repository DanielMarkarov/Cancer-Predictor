import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# keeps the plots in one place. calls image as static pngs
import matplotlib.pyplot as plt # side-stepping mpl backend
import matplotlib.gridspec as gridspec # subplots
#import mpld3 as mpl

#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

filepath ="/Users/danielmarkarov/Desktop/Polygence_Datasets/Brain_Tumor/Brain_Tumor.csv"
df = pd.read_csv (filepath,encoding = "ISO-8859-1")
df.head()
#df.drop('Unnamed: 32',axis=1,inplace=True)
## size of the dataframe
#len(df)
#df.diagnosis.unique()

#df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
# df.head()

# df.describe()
# plt.hist(df['diagnosis'])
# plt.title('Diagnosis (M=1 , B=0)')
# plt.show()

features_mean=list(df.columns[1:11])
# # split dataframe into two based on diagnosis
dfM=df[df['Mean'] ==1]
dfB=df[df['Mean'] ==0]

#Stack the data
plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
axes = axes.ravel()
for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(df[features_mean[idx]]) - min(df[features_mean[idx]]))/50
    ax.hist([dfM[features_mean[idx]],dfB[features_mean[idx]]], bins=np.arange(min(df[features_mean[idx]]), max(df[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, density = True, label=['M','B'],color=['r','g'])
    ax.legend(loc='upper right')
    ax.set_title(features_mean[idx])
plt.tight_layout()
plt.show()

traindf, testdf = train_test_split(df, test_size = 0.3)

#Generic function for making a classification model and accessing the performance.
# From AnalyticsVidhya tutorial
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])

  #Make predictions on training set:
  predictions = model.predict(data[predictors])

  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))
  #Perform k-fold cross-validation with 5 folds
  kf = KFold(n_splits=5)  
  error = []

  #***** Used to error on this loop that it couldn't iterate through a KFold
  #previously: for train, test in kf
  #changed it to iterate through df bc traindf/testdf already perform splits but unsure if that was the right move
  for train, test in kf.split(traindf):
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])

    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]

    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)

    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))

    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

#Creating a logistic regression model
  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome])
predictor_var = ['Mean', 'Variance', 'Standard Deviation', 'Entropy', 'Skewness',
                'Kurtosis', 'Contrast', 'Energy', 'ASM', 'Homogeneity', 'Dissimilarity',
                'Correlation', 'Coarseness']
outcome_var='diagnosis'
print("Logistic Regression Model:")
model=LogisticRegression()
classification_model(model,traindf,predictor_var,outcome_var)

#Decision Tree model
print("Decision Tree Model:")
predictor_var = ['Mean', 'Variance', 'Standard Deviation', 'Entropy', 'Skewness', 'Kurtosis', 'Contrast', 'Energy', 'ASM', 'Homogeneity', 'Dissimilarity', 'Correlation', 'Coarseness']
outcome_var='diagnosis'
model = DecisionTreeClassifier()
classification_model(model,traindf,predictor_var,outcome_var)

#Randome Forest model
print("Randome Forest Model:")
predictor_var = features_mean
model = RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)
classification_model(model, traindf,predictor_var,outcome_var)

#Create a series with feature importances from the randome forest
print("Feature Importance Series from Randome Forest:")
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(featimp)

# Randome forest classifier model using series data
print("Random Forest Classifier Model:")
predictor_var = ['Mean', 'Variance', 'Standard Deviation', 'Entropy', 'Skewness', 'Kurtosis', 'Contrast', 'Energy', 'ASM', 'Homogeneity', 'Dissimilarity', 'Correlation', 'Coarseness']
model = RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=2)
classification_model(model,traindf,predictor_var,outcome_var)




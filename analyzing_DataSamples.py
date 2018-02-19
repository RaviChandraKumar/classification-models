import pandas as pd
import matplotlib.pyplot as plot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

wine = pd.read_csv('wine.csv')
class_column = 'quality'

feature_columns = ['fixed acidity','volatile acidity','citric acid','residual sugar',\
                   'chlorides','free sulfur dioxide','total sulfur dioxide','density',\
                   'pH','sulphates','alcohol']

# =============================================================================
# print("Summary Statistics of each faeture class wise")
# for feature in feature_columns: 
#     byWineQuality = wine.groupby(class_column)
#     statsOfFeature = byWineQuality[feature].describe()
#     print("\n stats of the feature: ", feature, "\n", statsOfFeature)        
# 
# wine[wine.dtypes[(wine.dtypes=="float64")|(wine.dtypes=="int64")]
#                         .index.values].hist(figsize=[11,11])
# 
# =============================================================================

# =============================================================================
#predictions without pre-processing
wine_feature = wine[feature_columns]
wine_class = wine[class_column]
 
train_feature, test_feature, train_class, test_class = \
train_test_split(wine_feature, wine_class, stratify=wine_class, \
                 train_size=0.75, test_size=0.25)
 
knn = KNeighborsClassifier(n_neighbors=75, weights='distance', metric='canberra', p=2)
knn.fit(train_feature, train_class)
 
print("\n Training set accuracy using KNN: {:.3f}".format(knn.score(train_feature, train_class)))
 
print("\n Test set accuracy using KNN: {:.3f}".format(knn.score(test_feature, test_class)))
 
prediction = knn.predict(test_feature)
print("\n Confusion matrix using KNN:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

scores = []
scores = cross_val_score(knn, wine_feature, wine_class, cv=10)

print("\n Cross-validation scores: \n {}".format(scores))
print("\n Average cross-validation score using KNN: {:.3f}".format(scores.mean()))
 
# =============================================================================

# =============================================================================
# 
# #transformation on data - pre-pocessing
min_max=MinMaxScaler()

train_feature, test_feature, train_class, test_class = \
train_test_split(wine_feature, wine_class, stratify=wine_class, \
                 train_size=0.75, test_size=0.25)

wine_feature_minMax=min_max.fit_transform(wine_feature)
train_feature_minMax=min_max.fit_transform(train_feature)
test_feature_minMax=min_max.fit_transform(test_feature)
  
knn = KNeighborsClassifier(n_neighbors=75, weights='distance', metric='euclidean', p=2)
knn.fit(train_feature_minMax, train_class)

print("\n Training set accuracy using KNN: {:.3f}".format(knn.score(train_feature_minMax, train_class)))

print("\n Test set accuracy using KNN: {:.3f}".format(knn.score(test_feature_minMax, test_class)))

prediction = knn.predict(test_feature_minMax)
print("\n Confusion matrix using KNN:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

scores = []
scores = cross_val_score(knn, wine_feature_minMax, wine_class, cv=10)

print("\n Cross-validation scores: \n {}".format(scores))
print("\n Average cross-validation score using KNN: {:.3f}".format(scores.mean()))

# 
# =============================================================================

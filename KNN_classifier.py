import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

#read from the csv file and return a Pandas DataFrame.
wine = pd.read_csv('wine.csv')

# "quality" is the class attribute we are predicting which can have one of these values(4,5,6,7,8) => 5 classes.
class_column = 'quality'

#The attributes that we will use for classification
feature_columns = ['fixed acidity','volatile acidity','citric acid','residual sugar',\
                   'chlorides','free sulfur dioxide','total sulfur dioxide','density',\
                   'pH','sulphates','alcohol']

#split the data into features and class
wine_feature = wine[feature_columns]
wine_class = wine[class_column]

#splitting the data into a training and a test set (75% of the data for training and the rest 25% for testing)
train_feature, test_feature, train_class, test_class = \
    train_test_split(wine_feature, wine_class, stratify=wine_class, \
    train_size=0.75, test_size=0.25)

#build the model using K-nearest Neighbours classifier
knn = KNeighborsClassifier(n_neighbors=12, weights='distance', metric='minkowski', p=1)
knn.fit(train_feature, train_class)

#computing and printing the accuracy on the training set
print("\n Training set accuracy using KNN: {:.3f}".format(knn.score(train_feature, train_class)))

#computing and printing the accuracy on the test set
print("\n Test set accuracy using KNN: {:.3f}".format(knn.score(test_feature, test_class)))

#make predictions on the test data
prediction = knn.predict(test_feature)
#printing the Confusion Matrix along with 'All', as a result we have 6X6 Matrix instead of 5X5
print("\n Confusion matrix using KNN:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

#Array to capture the cross-validation accuracy for each of the 10-folds
scores = []
scores = cross_val_score(knn, wine_feature, wine_class, cv=10)

#printing cross-validation accuracy for each of the 10-folds
print("\n Cross-validation scores: \n {}".format(scores))
#printing cross-validation accuracy by computing the mean accuracy of the 10-folds
print("\n Average cross-validation score using KNN: {:.3f}".format(scores.mean()))

#save the training set into a CSV file named "train_data.csv"
train_class_df = pd.DataFrame(train_class,columns=[class_column])     
train_data_df = pd.merge(train_class_df, train_feature, left_index=True, right_index=True)
train_data_df.to_csv('train_data.csv', index=False)

#save the test set together with the predicted labels into another CSV file named "test_data.csv"
temp_df = pd.DataFrame(test_class,columns=[class_column])
temp_df['Predicted Pos']=pd.Series(prediction, index=temp_df.index)
test_data_df = pd.merge(temp_df, test_feature, left_index=True, right_index=True)
test_data_df.to_csv('test_data.csv', index=False)
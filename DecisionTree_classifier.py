import pandas as pd
import matplotlib.pyplot as plt

#import a subset of the modules
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

#read from the csv file and return a Pandas DataFrame.
# print the column names
# nba is the variable that has the whole data set imported from the Excel
print("\n", original_headers)
nba = pd.read_csv('NBAstats.csv')



##print the first three rows.
#print(nba[0:3])
## "(pos)" is the class attribute we are predicting. 
class_column = 'Pos'

##The dataset contains attributes such as player name and team name. 
##We know that they are not useful for classification and thus do not 
##include them as features. 
feature_columns = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', \
   '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', \
   'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']
#
##Pandas DataFrame allows you to select columns. 
##We use column selection to split the data into features and class. 
nba_feature = nba[feature_columns]
nba_class = nba[class_column]

train_feature, test_feature, train_class, test_class = \
    train_test_split(nba_feature, nba_class, stratify=nba_class, \
    train_size=0.75, test_size=0.25)
    
training_accuracy = []
test_accuracy = []

tree = DecisionTreeClassifier(max_depth=6, random_state=0)
tree.fit(train_feature, train_class)
print("Training set score: {:.3f}".format(tree.score(train_feature, train_class)))
print("Test set score: {:.3f}".format(tree.score(test_feature, test_class)))

prediction = tree.predict(test_feature)
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

train_class_df = pd.DataFrame(train_class,columns=[class_column])     
train_data_df = pd.merge(train_class_df, train_feature, left_index=True, right_index=True)
train_data_df.to_csv('train_data.csv', index=False)

temp_df = pd.DataFrame(test_class,columns=[class_column])
temp_df['Predicted Pos']=pd.Series(prediction, index=temp_df.index)
test_data_df = pd.merge(temp_df, test_feature, left_index=True, right_index=True)
test_data_df.to_csv('test_data.csv', index=False)
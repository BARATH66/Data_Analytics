import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image

Train_Dataset = pd.read_csv("/home/barath/Documents/DS/Python3/Loanpredtiction/train_ctrUa4K.csv")
Test_Dataset = pd.read_csv("/home/barath/Documents/DS/Python3/Loanpredtiction/test_lAUu6dG.csv")

del Train_Dataset['Loan_ID']
del Train_Dataset['Loan_Status']
del Test_Dataset['Loan_ID']

Dataset = pd.concat([Train_Dataset, Test_Dataset], axis=0,ignore_index=True)
Dataset.isnull().sum()    

NullFree_Dataset = pd.DataFrame()
def fill_na(dataset):
    for col in dataset.columns:
        if dataset[col].isnull().sum() == 0:
            NullFree_Dataset[dataset[col].name]= dataset[col]
        elif dataset[col].value_counts().count() <= 4:
            NullFree_Dataset[dataset[col].name] = dataset[col].fillna(dataset[col].value_counts().index[0])
        else:
            NullFree_Dataset[dataset[col].name] = dataset[col].fillna(np.floor(dataset[col].value_counts().mean()))
    return NullFree_Dataset
Dataset = fill_na(Dataset)
#Dataset

cat_vars = pd.DataFrame()
num_vars = pd.DataFrame()
fac_data = pd.DataFrame()
le = LabelEncoder()
def splitcatvars(dataset):
    for col in dataset.columns:
        if dataset[col].value_counts().count()  <=4:
            cat_vars[dataset[col].name] = dataset[col]
        else:
            num_vars[dataset[col].name] = dataset[col]
   
    return cat_vars, num_vars
cat_vars, num_vars = splitcatvars(Dataset)

def tonumeric(dataset):
    for col in dataset.columns:
        fac_data[dataset[col].name] = le.fit_transform(dataset[col])
    return fac_data
New_Dataset = tonumeric(cat_vars)
print("New_Dataset:",New_Dataset.shape,' ',"Num_vars:", num_vars.shape)

Dataframe = pd.concat([New_Dataset, num_vars],axis=1)
Trainset = Dataframe.iloc[:614,:]
Testset = Dataframe.iloc[614:,:]
print("Trainset:",Trainset.shape,' ',"Testset:", Testset.shape)

clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=3, max_features=None)

clf_train = clf.fit(Trainset, Target)

data_pic = tree.export_graphviz(clf_train, max_depth=3, feature_names=list(Trainset.columns), 
                                filled=True, label='root', rounded=True)

graph = pydotplus.graph_from_dot_data(data_pic)
Image(graph.create_png())

Train_pred = clf_train.predict(Trainset)

con_mat = confusion_matrix(Target, Train_pred)
acc_scr = accuracy_score(Target, Train_pred)
print("Confusion_Matrix:",con_mat,'\n',"Accuracy_Score:",acc_scr)

Test_pred = clf_train.predict(Testset)
acc_scr1 = cross_val_score(clf_train,Testset, Test_pred)
print("Accuracy_Score:",acc_scr1.mean())

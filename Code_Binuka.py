import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestClassifier
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout 
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score


def match_tables(transaction,labs):
    """ Function to combine two initial dataframe containing all transactions and fraudulent transactions
        create one dataframe with labels
            Args:
                transaction (dataframe): all transaction dataset
                labs (dataframe): fradulent dataset
            Return:
                dataframe: dataframe with labels """
    
    fraud = []
    labels = list(labs["eventId"])
    for index,row in transaction.iterrows():
        if row["eventId"] in labels:
            fraud.append(1)
        else:
            fraud.append(0)
    
    transaction["Fraud"] = fraud
        
    return transaction



def convert_to_categorical(categoricals,xcolumn):
    """ Function to convert categorical columns to category type
            Args:
                categoricals (list): list of categorical features
                xcolumn (dataframe): dataframe
            Return:
                dataframe: converted dataframe """
    
    for i in categoricals:
        xcolumn[i] = xcolumn[i].astype("category")
    return xcolumn


def convert_to_timedate(df):
    """ Function to convert transaction time column to Datetime type and order by date time
            Args:
                df (dataframe): dataframe
            Return:
                dataframe: converted dataframe """
    
    df["transactionTime"] = pd.to_datetime(df["transactionTime"], infer_datetime_format=True)
    df.sort_values(by='transactionTime', inplace=True)
    return df


def calculate_time_to_lastTransaction(df):
    """ Function to find the time between transactions
            Args:
                df (dataframe): dataframe
            Return:
                list: list of time between transactions """
    
    #Dictionary to store account numbers
    dic = {}
    #list to store time between transactions
    time_since_last_transaction = []

    #using a for loop, previous transaction by the same account number was subtracted by the current one and added to a list as total seconds. 
    #If no previous transaction for the account number, 0 seconds was added to the list
    for index,row in df.iterrows():
        if row["accountNumber"] not in dic.keys():
            dic[row["accountNumber"]] = row["transactionTime"]
            time_since_last_transaction.append(0)
        else:
            time = (row["transactionTime"] - dic[row["accountNumber"]]).total_seconds()
            time_since_last_transaction.append(time)
            dic[row["accountNumber"]] = row["transactionTime"]
            
    return time_since_last_transaction


def scale_numeric_features(df):
    """ Function to scale numeric features
            Args:
                df (dataframe): dataframe
            Return:
                dataframe: scaled dataframe """
    
    sc = StandardScaler()
    cols = ['transactionAmount', 'availableCash',"time since last transaction in seconds"]
    df[cols] = sc.fit_transform(df[cols])
    return df

def pca(X):
    """ Function to condcut PCA on the dataframe
            Args:
                X (dataframe): dataframe
            Return:
                dataframe: PCA dataframe """
    
    pca = PCA(0.90)
    X_pca = pca.fit_transform(X)
    X_pca = pca.transform(X)
    return X_pca


def Evaluation_metrics(y_test, y_pred):
    """ Function to calculate evaluation metrics
            Args:
                y_test(dataframe): test target dataframe
                y_pred(dataframe): predicted target dataframe
    """
    
    # Model Accuracy Score 
    print(f"Accuracy \n{metrics.accuracy_score(y_test, y_pred)*100} %\n" )

    # Model Recall Score 
    print(f"Recall  \n{metrics.recall_score(y_test, y_pred)*100} %\n" )

    # Model Confusion Matrix
    print(f"Confusion Matrix\n{metrics.confusion_matrix(y_test, y_pred)}\n" )

    # Model Classification Report
    print("classification report\n\n",metrics.classification_report(y_test, y_pred))


def isolation_forest(df,y):
    """ Function to conduct isolation forest algorithm
            Args:
                df (dataframe): dataframe
                y (dataframe): target dataframe
    """
    
    ifa = df[["posEntryMode","transactionAmount","availableCash","merchantCountry","mcc","time since last transaction in seconds"]]
    model=IsolationForest(n_estimators=100,max_samples=len(ifa),contamination=0.007431,random_state=402)

    model.fit(ifa)

    print(model.get_params())
    scores_prediction = model.decision_function(ifa)
    y_pred = model.predict(ifa)
    
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    print("\n\n Isolation Forest Results \n")
    # Model Accuracy Score 
    print(f"Accuracy \n{metrics.accuracy_score(y, y_pred)*100} %\n" )
    
    # Model Recall Score 
    print(f"Recall  \n{metrics.recall_score(y, y_pred)*100} %\n" )

    # Model Confusion Matrix
    print(f"Confusion Matrix\n{metrics.confusion_matrix(y, y_pred)}\n" )

    # Model Classification Report
    print("classification report\n\n",metrics.classification_report(y, y_pred))



def main(model,isf): 
    """ Main Function
            Args:
                model (dataframe): default= lr(logistic regression),rf(random forest),svm(SVM),
                isf (dataframe): default= n(No isolation forest), y(yes for isolation forest) 
    """
    
    #Read Transactions dataset
    df = pd.read_csv("./data-new/transactions_obf.csv")
    #Read Labels Dataset
    labs = pd.read_csv("./data-new/labels_obf.csv")
    df = match_tables(df,labs)
    #Intial EDA and Data Visulaization was done using Jupyter Notebook
    #Categorical Columns
    categoricals = ["accountNumber","merchantId","mcc","merchantCountry","merchantZip","posEntryMode","Fraud"]
    #Convert categorical columns to categorical type
    df = convert_to_categorical(categoricals=categoricals,xcolumn=df)
    #Convert transaction time column to datatime
    df = convert_to_timedate(df)
    #Calulcate time between transactions
    time_since_last_transaction = calculate_time_to_lastTransaction(df)
    #Insert the time between transactions list to dataframe 
    df.insert(1, "time since last transaction in seconds", time_since_last_transaction)
    #Scale Numeric Features
    df = scale_numeric_features(df)
    #Seperate X and Y
    X = df[["posEntryMode","transactionAmount","availableCash","merchantCountry","accountNumber","mcc","time since last transaction in seconds"]]
    y = df["Fraud"]
    #One hot enocde categorical features
    X = pd.get_dummies(X,columns=["posEntryMode","merchantCountry","accountNumber","mcc"])
    #PCA to reduce dimensionality
    X_pca = pca(X)
    #Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y,
                                                        stratify=y, 
                                                        test_size=0.3,
                                                       random_state=402)
    #SMOTE to upsample minority fraud class to equal majority non fraud class(balance classes)
    sm = SMOTE(random_state=42)
    X_train,y_train= sm.fit_resample(X_train, y_train)


    #Model Selection
    if model == "rf":
        clf = RandomForestClassifier(n_estimators=100,random_state=10)
        print("Random Forest Results \n")
    elif model == "svm":
        clf = svm.SVC(kernel='linear')
        print("SVM Results \n")
    else:
        clf = LogisticRegression(solver='lbfgs', max_iter=1000,random_state=0) 
        print("Logistic Regression Results \n")

    #Model Training
    clf.fit(X_train, y_train)
    #Model Predicting
    y_pred = clf.predict(X_test)
    #Evaluation of model
    Evaluation_metrics(y_test, y_pred)

    #Select isolation forest or not
    if isf == "y":
        isolation_forest(df,y)


if __name__ == "__main__":
    #Type in model 'rf' for  random forest, 'svm' for SVM, 'lr' for logisitic regression (default = logistic regression) 
    #Type in isf 'y' for Isolation forest or 'n' for no Isolation forest (default = 'n' No Isolation Forest)
    main(model="lr",isf="n")
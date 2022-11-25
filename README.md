# BankFraudDetection

The two initial spreadsheets containing all transactions and fraudulent transaction were combined using a python function to create one dataframe with labels. 
Initial Exploratary data analysis (finding missing values and duplicates) and Data Visulaisation using countplots, boxplots and barcharts were done for the dataset.
There were 117746 non fraud data and 875 fraud data. Thus this dataset is unbalanced.
Since Merchant zip had 23005 missing values, this feature was dropped from the dataframe and not considered for the model. There were no other missing or duplicate values.
Then Categorical features were made as category type in the data frame and Transaction time feature was converted to Data Time type and the whole dataset was then ordered by Transaction time.
Then Feature Engineering was conducted to find the time between transactions. For this a dictionary was used to store the account numbers and using a for loop, previous transaction by the same account number was subtracted by the current one and added to a list as total seconds. If no previous transaction for the account number, 0 seconds was added to the list.
Then all the numeric features were scaled using standard scalar function in python.
Then X and Y columns were separated with X representing the features and Y representing the labels.
Then Categorical features were one hot encoded producing 1221 features.
As dimensionality was high, PCA was done on the dataset to reduce dimensionality.
A Stratified train-test split based on fraud or non fraud was done due to the imbalance of classes. The train set 70% and test set had 30% of the data.
Then SMOTE was conducted on the dataset to upsample the minority class of fraud records to match the non fraud records. This makes the model to be trained on equal fraud and non fraud data.
Then 3 traditional supervised classification machine learning models of Random Forest, SVM and Logistic regression were trained and tested.
Recall was selected as the evaluation metric because finding the true positives(frauds) is the most important task for bank fraud detection. In addition accuracy, precision, F1 score and confusion matrix was also produced.
An Isolation Forest model was also created and this is unsupervised anomaly detection algorithm and it was also trained and tested.
Out of the 4 models, Logistic regression model was the best model as it was able to correctly identify 223 fraudulent activity from 263 total fraudulent activity from and produced a recall score of 0.84 or 84.7% and accuracy of 85.9%. 




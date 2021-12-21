import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from imblearn import over_sampling


# ------------------------------------------------------------------------------
# Helper function to report Accuracy and F1-score
# ------------------------------------------------------------------------------
def metrics(actual_labels, predictions):
    print("Accuracy: {:.5f}".format(accuracy_score(actual_labels, predictions)))
    print("F1-score: {:.5f}".format(f1_score(actual_labels, predictions)))


# ------------------------------------------------------------------------------
# Visualization of given creditcard farud data set
# ------------------------------------------------------------------------------
df = pd.read_csv('creditcard.csv')
labels = ["Normal", "Fraud"]
count_classes = df.value_counts(df['Class'], sort= True)
count_classes.plot(kind = "bar", rot = 0)
plt.title("Visualization of Class of Credit card Fraud")
plt.ylabel("Count")
plt.xticks(range(2), labels)
plt.show()


# ------------------------------------------------------------------------------
# Normalize data
# ------------------------------------------------------------------------------
rs = RobustScaler()
df['scaled_amount'] = rs.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rs.fit_transform(df['Time'].values.reshape(-1,1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)
scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']
df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(0, 'scaled_time', scaled_time)

## splite features and labels
x = df.drop(["Class"], axis= 1)
y = df["Class"]


# # # ------------------------------------------------------------------------------
# # Supervised learning with Random Forest classifier
# # ------------------------------------------------------------------------------
## Splite the data into training and test data sets
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size= 0.3, random_state= 42)

## Apply Decision Tree model with unbalanced data
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

print('Evaluation of Random Forest Model Before SMOT:')
## Call helper function to get the result
metrics(y_test, y_pred.round())


## Apply Synthetic Minority Oversampling Technique to balance the data points
## so that Class 1 and 0 has a 1:1 ratio
sm=over_sampling.SMOTE(random_state=2)
x_train_s, y_train_s = sm.fit_resample(x_train, y_train)

## Apply Decision Tree model
rf = RandomForestClassifier()
rf.fit(x_train_s, y_train_s)
y_pred = rf.predict(x_test)

print('Evaluation of Random Forest Model After SMOT:')
## Call helper function to get the result
metrics(y_test, y_pred.round())


# ------------------------------------------------------------------------------
# Unsupervised learning with Isolation Forest
# ------------------------------------------------------------------------------
## set n_estimators as default 100
## set contamination as the fraud transaction ratio
isolation_forest = IsolationForest(n_estimators=100, n_jobs=-1, contamination=0.00172, max_samples=0.25)
isolation_forest.fit(x_train)
y_pred = isolation_forest.predict(x_test)

## 0 is normal transaction, 1 is fraud transaction
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

print('Evaluation of Isolation Forest Model:')
## Call helper function to get the result
metrics(y_test, y_pred.round())
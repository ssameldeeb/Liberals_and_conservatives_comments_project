import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix , accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("file_name.csv")

print(data.shape)
print(data.columns.values)
print(data.isnull().sum())
print(data.info())
print(data.dtypes)
print(data.head())

data = data.drop(["Text","Id","Title"], axis=1)
print(data.isnull().sum())

print(data["Political Lean"].unique())
print(data["Political Lean"].value_counts())

La = LabelEncoder()
data["Political Lean"] = La.fit_transform(data["Political Lean"])
data["Subreddit"] = La.fit_transform(data["Subreddit"])
data["URL"] = La.fit_transform(data["URL"])

print(data.dtypes)
print(data.head(3))

# Logistic_model

x = data.drop("Political Lean",axis=1)
y = data["Political Lean"]

ss = StandardScaler()
x = ss.fit_transform(x)
print(x[:5])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle =True)
print(X_train.shape)

Lo = LogisticRegression()
Lo.fit(X_train, y_train)

print("_"*100)
print(Lo.score(X_train, y_train))
print(Lo.score(X_test, y_test))
print("_"*100)


# print("_"*150)
# for x in range(2,20):
#     Dt = DecisionTreeClassifier(max_depth=x,random_state=33)
#     Dt.fit(X_train, y_train)

#     print("x = ", x)
#     print(Dt.score(X_train, y_train))
#     print(Dt.score(X_test, y_test))
#     print("_"*100)



# LinearRegression_model

# x = data.drop("Score",axis=1)
# y = data["Score"]

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle =True)
# print(X_train.shape)

# Li = LinearRegression()
# Li.fit(X_train, y_train)

# print("_"*100)
# print(Li.score(X_train, y_train))
# print(Li.score(X_test, y_test))
# print("_"*100)


# DecisionTreeClassifier_model
print("_"*150)
Dt = DecisionTreeClassifier(max_depth=15,random_state=33)
Dt.fit(X_train, y_train)

print(Dt.score(X_train, y_train))
print(Dt.score(X_test, y_test))
print("_"*100)
y_pred = Dt.predict(X_test)

# confusion_matrix
Cm = confusion_matrix(y_test,y_pred)
print(Cm)
sns.heatmap(Cm,annot=True, fmt="d", cmap="magma")
plt.show()

print("_"*100)

# accuracy_score
As = accuracy_score(y_test,y_pred)
print(As)


# The autput result
result = pd.DataFrame({"y_test":y_test, "y_pred":y_pred})
# result.to_csv("The autput.csv",index=False)
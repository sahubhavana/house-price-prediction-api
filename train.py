
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
import pandas as pd
import joblib
print("loadig dataset")
data=pd.read_csv("house_prediction_dataset.csv")

X = data.drop(["Price", "House_ID"], axis=1)
y = data["Price"]

print(f"total records:{X.shape[0]}")
X = pd.get_dummies(X, drop_first=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model=RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train,y_train)


y_pred=model.predict(X_test)
mse=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
joblib.dump(mse, "mse.joblib")

print(f"average error:${mse*100000}")


# # now model can save in joblib file ##
joblib.dump(model,"House-Prediction.joblib")
joblib.dump(list(X.columns),"house_feature.joblib")


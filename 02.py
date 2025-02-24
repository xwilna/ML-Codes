import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd


X, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False)
X = MinMaxScaler().fit_transform(X)

model = SVC(verbose=True)
model.fit(X,y)

y_pred = model.predict(X)
print(classification_report(y, y_pred))
print(model.score(X, y) * 100)

dataset = np.load("/content/digits.npz")
new_X = dataset["data"]
new_y = dataset["target"]

new_X = MinMaxScaler().fit_transform(new_X)

y_new_pred = model.predict(new_X)

print(f"predictions on new data: {y_new_pred[:10]}")

with open("model.pkl", "wb") as f:
  pickle.dump(model, f)

results_df1_new = pd.DataFrame({
    'true label' : y,
    'predicted label' : y_pred
})
results_df1_new['correct predict'] = np.where(results_df1_new['true label'] == results_df1_new['predicted label'],1,0)
sum_correct = results_df1_new['correct predict'].sum()
results_df1_new['toltal','correct predict'] = sum_correct
results_df1_new.to_csv('mnist_predict_1',index=False)

results_df2_new = pd.DataFrame({
    'true label' : new_y,
    'predicted label' : y_new_pred
})
results_df2_new['correct predict'] = np.where(results_df2_new['true label'] == results_df2_new['predicted label'],1,0)
sum_correct2 = results_df2_new['correct predict'].sum()
results_df2_new['toltal','correct predict'] = sum_correct2
results_df2_new.to_csv('mnist_predict_1',index=False)

with open("model_new.pkl", "wb") as f:
     pickle.dump(model, f)

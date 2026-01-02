# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# %%
df=pd.read_csv("data/iris.csv")

# %%
training_df=df.sample(frac=0.75,random_state=42)
testing_df=df.drop(training_df.index).reset_index(drop=True)
training_df=training_df.reset_index(drop=True)

# %%
model=RandomForestClassifier()
model.fit(training_df.drop("species",axis=1),training_df["species"])

# %%
prediction=model.predict(testing_df.drop("species",axis=1))
prediction_comparison=list(zip(prediction,testing_df["species"]))
prediction_df=pd.DataFrame(prediction_comparison,columns=["Predicted Species","Actual Species"])
print(prediction_df.iloc[:10])

# %%

accuracy=np.mean(prediction==testing_df["species"])
print(f"Model accuracy: {accuracy*100:.2f}%")

# %%




import  os 
import pandas as pd 
import numpy as np
data = pd.read_csv('./csv/mol_interaction_energy.csv',sep=',').iloc[0:48236,[1,2]]
print("--------------------Original data---------------------")
print(data.head())
# data.columns = ["molecule_id","SMILES","TARGET"]
data.columns = ["SMILES","TARGET"]

train_fraction = 0.8
train_data = data.sample(frac=train_fraction,random_state=1)
train_data.to_csv("./csv/train.csv",index=False)
test_data = data.drop(train_data.index)
test_data.to_csv("./csv/test.csv",index=False)

train_y = np.array(train_data["TARGET"].values.tolist())
test_y = np.array(test_data["TARGET"].values.tolist())


results = {}


import matplotlib.pyplot as plt 
import seaborn as sns
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 15}
plt.hist(train_data["TARGET"], bins=20, label="Train Data")
plt.hist(test_data["TARGET"], bins=20, label="Test Data")
plt.ylabel("Count", fontdict=font)
plt.xlabel("energy", fontdict=font)
plt.legend()
# plt.show()


import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from unimol_tools import UniMolRepr


def calculate_unimol_qsar_repr(data):
    clf = UniMolRepr(data_type='molecule', remove_hs=False)
    smiles_list = data["SMILES"].tolist()  
    repr_dict = clf.get_repr(smiles_list,return_atomic_reprs=True)  
    unimol_repr_list = np.array(repr_dict['cls_repr'])
    unimol_repr = repr_dict['cls_repr']
    return unimol_repr



train_data["unimol_qsar_mr"] = calculate_unimol_qsar_repr(train_data)  
test_data["unimol_qsar_mr"] = calculate_unimol_qsar_repr(test_data)


print(train_data["2dqsar_mr"].iloc[:1].values.tolist())


import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

train_x = np.array(train_data["2dqsar_mr"].values.tolist())
train_y = np.array(train_data["TARGET"].values.tolist())
test_x = np.array(test_data["2dqsar_mr"].values.tolist())
test_y = np.array(test_data["TARGET"].values.tolist())
print(train_x.shape)

train_x = np.reshape(train_x, (train_x.shape[0], -1))
test_x = np.reshape(test_x, (test_x.shape[0], -1))

regressors = [
    ("Linear", LinearRegression(),{}), 
    ("RR", Ridge(random_state=42),{'alpha': [0.1, 1, 10]}), 
    ("Lasso", Lasso(random_state=42),{'alpha': [0.1, 1, 10]}), 
    ("ER", ElasticNet(random_state=42),{'alpha': [0.1, 1, 10],'l1_ratio': [0.1,0.5,0.9]}), 
    ("SV", SVR(),{'C': [0.1,1,10]}),  
    ("K-NN", KNeighborsRegressor(),{'n_neighbors': [3,5,7]}),  
    ("DT", DecisionTreeRegressor(random_state=42),{'max_depth': [3,5,7]}),  
    ("RF", RandomForestRegressor(random_state=42),{'n_estimators': [50,100,200]}), 
    ("GB", GradientBoostingRegressor(random_state=42),{'n_estimators': [50,100,200]}), 
    ("XGB", XGBRegressor(random_state=42),{'n_estimators': [50,100,200]}), 
    ("LGBM", LGBMRegressor(random_state=42),{'num_leaves': [31,63,127]}), 
    ("MLP", MLPRegressor(
        hidden_layer_sizes=(128,64,32),
        learning_rate_init=0.0001,
        activation='relu', solver='adam', 
        max_iter=10000, random_state=42),{'hidden_layer_sizes': [(64,),(128,64,32)]}),
]

with open('./csv/performance_regression.txt','a') as f:
    f.write(f"===============================================\n")

for name, regressor, param_grid in regressors:
    grid_search = GridSearchCV(regressor,param_grid,cv=5) 
    grid_search.fit(train_x,train_y)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # regressor.fit(train_x, train_y)
    best_model.fit(train_x,train_y)
    
    pred_train_y = best_model.predict(train_x)
    pred_test_y = best_model.predict(test_x)
    
    train_data[f"{name}_pred"] = pred_train_y
    test_data[f"{name}_pred"] = pred_test_y
    
    mse = mean_squared_error(test_y, pred_test_y)
    mae = mean_absolute_error(test_y,pred_test_y)
    r2 = r2_score(test_y,pred_test_y)
    se = abs(test_y - pred_test_y)
    results[f"{name}"] = {"MSE": mse, "MAE": mae, "R^2": r2, "error": se}
    print(f"[unimol-QSAR][{name}]\tMSE:{mse:.4f}\tMAE:{mae:.4f}\tR^2:{r2:.4f}")

       
    with open('/vol8/home/mmx/AI-for-M/machine/csv/performance_regression.txt','a') as f:
        f.write(f"[unimol-QSAR][{name}]\tMSE:{mse:.4f}\tMAE:{mae:.4f}\tR^2:{r2:.4f}\n")


from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


residuals_data = []
for name, result in results.items():
    if name.startswith("unimol-QSAR"):
        model_residuals = pd.DataFrame({"Model": name, "Error": result["error"]})
        residuals_data.append(model_residuals)

residuals_df = pd.concat(residuals_data, ignore_index=True)
residuals_df.sort_values(by="Error", ascending=True, inplace=True)
model_order = residuals_df.groupby("Model")["Error"].median().sort_values(ascending=True).index


plt.figure(figsize=(10, 7), dpi=300)
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 15}
sns.boxplot(y="Model", x="Error", data=residuals_df, order=model_order)
plt.yticks(rotation=45,fontsize=10)
plt.xlabel("Abs Error", fontdict=font)
plt.ylabel("Models", fontdict=font)
plt.savefig("/vol8/home/mmx/AI-for-M/machine/figure/cancha_plot.png",dpi=300)


for name, regressor, param_grid in regressors:
    model_name = name
    pred_values = test_data[f"{model_name}_pred"]
    actual_values = test_y

    plt.figure(figsize=(8,6))
    plt.scatter(actual_values,pred_values,color='blue',alpha=0.5)

   
    z = np.polyfit(actual_values,pred_values,1)
    p = np.poly1d(z)
    plt.plot(actual_values,p(actual_values),color='red')

    plt.title(f"classification for {model_name}")
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.savefig(f"./figure/{model_name}_regression.png",dpi=300)


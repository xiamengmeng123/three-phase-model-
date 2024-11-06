import pandas as pd 
from sklearn.linear_model import LogisticRegression
import joblib 
from unimol_tools import UniMolRepr

def calculate_unimol_qsar_repr(data):
    clf = UniMolRepr(data_type='molecule', remove_hs=False)
    smiles_list = data["SMILES"].tolist()  
    repr_dict = clf.get_repr(smiles_list,return_atomic_reprs=True)  
    unimol_repr = repr_dict['cls_repr']
    return unimol_repr

data = pd.read_csv('./csv/million_dataset.csv')
X = calculate_unimol_qsar_repr(data)


loaded_model = joblib.load('./lrm/logistic_regression_model.pkl')

Y_pred = loaded_model.predict(X)

selected_data = data[Y_pred == 1]
selected_data.to_csv('./csv/erfenlei_selected_million.csv',index=False,header=None)

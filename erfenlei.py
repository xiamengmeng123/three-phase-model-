#########二分类任务
import pandas as pd 
import numpy as np  
from sklearn.cluster import DBSCAN 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score
import matplotlib.pyplot as plt  
from unimol_tools import UniMolRepr
import joblib
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score,precision_recall_curve,auc

def calculate_unimol_qsar_repr(data):
    clf = UniMolRepr(data_type='molecule', remove_hs=False)
    smiles_list = data["SMILES"].tolist()  
    repr_dict = clf.get_repr(smiles_list,return_atomic_reprs=True)  
    unimol_repr = repr_dict['cls_repr']
    return unimol_repr


data = pd.read_csv('./csv/mol_interaction_energy.csv',header=None)
data.columns=["molecule_id","SMILES","TARGET"]
print(data.head())

sorted_data = data.sort_values(by='TARGET',ascending=False)
sorted_data = sorted_data.drop_duplicates(subset='molecule_id')
sorted_data = sorted_data.reindex(columns=data.columns)
sorted_data.to_csv('./csv/sorted_mol_interaction_energy.csv',header=None,index=False)


breakpoint_value = sorted_data.iloc[48236]['TARGET']  
print(breakpoint_value)
plt.figure(figsize=(10,6))
plt.hist(data['TARGET'],bins=50)
plt.axvline(x=breakpoint_value,color='r',linestyle="--")
plt.savefig('./figure/breakpoint.png',dpi=300)


threshold = breakpoint_value


sorted_data['TARGET_CLASS'] = sorted_data['TARGET'].apply(lambda x: 1 if x >= threshold else 0)


X = calculate_unimol_qsar_repr(sorted_data)
y = sorted_data['TARGET_CLASS']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


model = LogisticRegression()
model.fit(X_train,y_train)


joblib.dump(model,'./lrm/logistic_regression_model.pkl')


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)


conf_matrix = confusion_matrix(y_test,y_pred)
print("confusion matrix",conf_matrix)


fpr,tpr,thresholds = roc_curve(y_test,model.predict_proba(X_test)[:,1])
roc_auc = roc_auc_score(y_test,model.predict_proba(X_test)[:,1])

plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,color='darkorange',lw=2,label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
plt.xlabel('Fales Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.savefig('./figure/erfenlei.png',dpi=300)





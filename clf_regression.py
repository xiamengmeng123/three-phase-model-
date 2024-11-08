import pandas as pd
import argparse
from unimol_tools import MolTrain,MolPredict
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt 
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int, help="Data segmentation point")
args = parser.parse_args()

data = pd.read_csv('./csv/mol_interaction_energy.csv',sep=',').iloc[0:args.num,[1,2]]
print(data.head())
data.columns = ["SMILES","TARGET"]


train_fraction = 0.8
train_data = data.sample(frac=train_fraction,random_state=1)
train_data.to_csv("./csv/train.csv",index=False,header=['SMILES','TARGET'])
test_data = data.drop(train_data.index)
test_data.to_csv("./csv/test.csv",index=False,header=['SMILES','TARGET'])


clf = MolTrain(task='regression',
                data_type='molecule',
                epochs=100,
                learning_rate=0.0001,
                batch_size=32,
                early_stopping=10,
                metrics='mse',
                split='random',
                save_path='./clf',
                )


clf.fit('./csv/train.csv')


clf = MolPredict(load_model='./clf_interaction')
test_path = './csv/test.csv'
test_pred = clf.predict(test_path)


df = pd.read_csv(test_path,header='infer')
test_target = df['TARGET'].values


residuals = np.abs(test_target - test_pred.flatten())
threshold = np.percentile(residuals,95)   
print(threshold)
noise_indices = np.where(residuals > threshold)[0]


cleaned_test_data = test_data.drop(test_data.index[noise_indices])
cleaned_test_data.to_csv("./csv/test_cleaned.csv",index=False)

test_pred = clf.predict('./csv/test_cleaned.csv')
test_clean_path = './csv/test_cleaned.csv'


df = pd.read_csv(test_clean_path,header='infer')
test_target = df['TARGET'].values


rmse_test = np.sqrt(mean_squared_error(test_target,test_pred.flatten()))
R2_test = r2_score(test_target,test_pred.flatten())

fig, ax = plt.subplots(figsize=(5,5),dpi=300)
xmin = min(test_pred.flatten().min(),test_target.min())
xmax = max(test_pred.flatten().max(),test_target.max())
ymin = xmin
ymax = xmax

ax.scatter(test_target,test_pred.flatten(),alpha=0.2,s=10,c='red',label='Test')
# ax.scatter(test_target[noise_indices],test_pred.flatten()[noise_indices],c='blue',label='Noise Ponints')

ax.text(0.6,0.11,"RMSE (Test) = " + "%.3f"%(rmse_test),fontsize=10,transform=ax.transAxes)
ax.text(0.6,0.07,"R$^{2}$ (Test) = " + "%.3f"%(R2_test),fontsize=10,transform=ax.transAxes)

plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

ax.set_xlabel('target')
ax.set_ylabel('predict',labelpad=-10)

_ = ax.plot([xmin,xmax],[ymin,ymax],c='k',ls='--')
ax.legend(loc='upper left')

plt.savefig('./figure/clf_predict_target.png',dpi=300)

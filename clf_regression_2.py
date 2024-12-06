import pandas as pd
from unimol_tools import MolTrain,MolPredict
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt 
import numpy as np 
import joblib



data = pd.read_csv('./csv/example_free_energy.csv',sep=',').iloc[:,[1,2]]
print(data.head())
data.columns = ["SMILES","TARGET"]

train_fraction = 0.8
train_data = data.sample(frac=train_fraction,random_state=1)
train_data.to_csv("./csv/train_fep_2.csv",index=False)
test_data = data.drop(train_data.index)
test_data.to_csv("./csv/test_fep_2.csv",index=False)

##训练模型初始化
clf = MolTrain(task='regression',
                data_type='molecule',
                epochs=100,
                learning_rate=0.0001,
                batch_size=32,
                early_stopping=10,
                metrics='mse',
                split='random',
                save_path='./clf_fep',  
                )

clf.fit('./csv/train_fep_2.csv')

clf = MolPredict(load_model='./clf_fep')  
test_path = './csv/test_fep_2.csv'
test_pred = clf.predict(test_path)

df = pd.read_csv(test_path,header='infer')
test_target = df['TARGET'].values

residuals = np.abs(test_target - test_pred.flatten())
threshold = np.percentile(residuals,95)   
print(threshold)
noise_indices = np.where(residuals > threshold)[0]

cleaned_test_data = test_data.drop(test_data.index[noise_indices])
cleaned_test_data.to_csv("./csv/test_fep_2_cleaned.csv",index=False)

test_pred = clf.predict('./csv/test_fep_2_cleaned.csv')
test_clean_path = './csv/test_fep_2_cleaned.csv'

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
ax.set_ylabel('predict',labelpad=0,fontsize=10)
# ax.set_title('free energy of target-predict',fontsize=14)   #!!!

_ = ax.plot([xmin,xmax],[ymin,ymax],c='k',ls='--')
ax.legend(loc='upper left')

plt.savefig('./figure/clf_predict_target_2.png',dpi=300) 

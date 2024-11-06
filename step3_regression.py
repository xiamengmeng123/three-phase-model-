import pandas as pd 
from unimol_tools import MolTrain,MolPredict

validation_path = ('./csv/step2_prediction_results.predict.10000.csv')
data = pd.read_csv(validation_path,usecols=['molecule_id','SMILES','pred']) #'molecule_id',

filtered_data = data[data['pred'] > 200]
filter_path = './csv/filtered_data.csv'
filtered_data.to_csv(filter_path,index=False)

clf = MolPredict(load_model='./clf_fep')

filtered_data['fep'] = clf.predict(filter_path)
filtered_data = filtered_data.sort_values(by='fep').reset_index(drop=True)
clf.save_predict(data=filtered_data,dir='./csv/',prefix='step3_prediction_results')
import pandas as pd 
from unimol_tools import MolTrain,MolPredict

validation_path = ('./csv/erfenlei_selected_10000.csv')
data = pd.read_csv(validation_path,usecols=['molecule_id','SMILES'])#'molecule_id',
data.columns = ["molecule_id","SMILES"]


clf = MolPredict(load_model='./clf_interaction')

validation_pred = clf.predict(validation_path)
data['pred'] = validation_pred
clf.save_predict(data=data,dir='./csv/',prefix='step2_prediction_results')





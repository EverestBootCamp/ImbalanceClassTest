Task objective:
Use TPOT method to find optimal pipiline for given dataset
Deliverable:
A new class which can be run using specific TPOT configs and desired parameters

Instructions to run tpot

1.makesure in the dataset target class lebel column name is 'Class'
2.The parameter Fullyqualified_op_filename with path along with additional '\' is required to generate python file for winning pipeline. for example: Fullyqualified_op_filename= "C:\\Users\\SG\\Desktop\\lantern\\local_notebook\\winning_pipeline.py"
3.tpot_maxtime parameter is max time to run tpot default is set to 720 mins and can be modified as needed
4. Example to call tpot:

t=Tpot(datafile= "data_transformed.csv")
from datetime import datetime
start=datetime.now()
t.apply_tpot(Fullyqualified_op_filename='C:\\Users\\SG\\Desktop\\lantern\\local_notebook\\wining2.py',tpot_config='C:\\Users\\SG\\Desktop\\lantern\\local_notebook\\ImbalanceClassTest\\notebooks\\tpot_config.py',tpot_maxtime=10)

print (datetime.now()-start)
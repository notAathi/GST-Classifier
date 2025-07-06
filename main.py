from src.utils import *
from src.loader import Model
import pandas as pd

x_test=pd.read_csv('./Test/X_Test_Data_Input.csv',index_col='ID')
y_test=pd.read_csv('./Test/Y_Test_Data_Target.csv',index_col='ID')

model=Model('./model/model.pkl')
model.metrics(x_test,y_test)
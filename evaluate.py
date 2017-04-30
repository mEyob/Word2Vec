import model
import numpy as np
import gensim
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
import os

arch = [0,1] 
ws = [5, 10] 
ds = [0.001,0.0001]
fv = [200, 400]

df_final = pd.DataFrame()

for arch_item in arch:
    for ws_item in ws:
        for ds_item in ds:
            for fv_item in fv:
				
                if arch_item == 0:
                    alg = "CBOW"
                else :
                    alg = "SkipGram"
                model_name = alg+'_ws'+ str(ws_item) + '_ds'+ str(ds_item) + '_ft' + str(fv_item)
                m = model.load_and_evaluate(model_name) 
                df = pd.DataFrame([m.analogical])
                df = df.join(pd.DataFrame([m.noun_classification]))
		
                time = {"Train_time": m.train_time}
                df = df.join(pd.DataFrame([time]))
                df_final = df_final.append(df,ignore_index=True)
                df_final.to_csv(os.getcwd()+"/results/Evaluation_result.csv")
                m.nearest_words.to_csv(os.getcwd()+"/results/"+model_name+".csv")
          
                


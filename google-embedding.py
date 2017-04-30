import model
import numpy as np
import gensim
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
import os

df_final = pd.DataFrame()
model_name = 'GoogleNews-vectors-negative300.bin.gz'
m = model.load_and_evaluate(model_name, google=True) 
df = pd.DataFrame([m.analogical])
df = df.join(pd.DataFrame([m.noun_classification]))
		
#time = {"Train_time": m.train_time}
#df = df.join(pd.DataFrame([time]))
df_final = df_final.append(df,ignore_index=True)
df_final.to_csv(os.getcwd()+"/results/google-evaluation_result.csv")
m.nearest_words.to_csv(os.getcwd()+"/results/google-nearest_words.csv")
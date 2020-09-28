import pyprind
import pandas as pd
import os
#Positive review in trainset
path = 'C:\\Python Software\\Emotional Analysis\\movie\\train\\pos'
files_0 = os.listdir(path) #Read the following txt documents in this file
df_0 = pd.DataFrame()
for file in files_0: 
    position_0 = path+'\\'+ file          
    with open(position_0, "r",encoding='utf-8') as f:
        data_0 = f.read()   #Read Txt
        df_0 = df_0.append([[data_0, 1]],ignore_index= True)
df_0.columns = ['Message','label']
#Negative review in trainset
path_1 = 'C:\\Python Software\\Emotional Analysis\\movie\\train\\neg'
files_1= os.listdir(path_1) #Read the following txt documents in this file
df_1 = pd.DataFrame()
for a in files_1: 
    position_1 = path_1+'\\'+ a        
    with open(position_1, "r",encoding='utf-8') as f_1:
        data_1 = f_1.read()   #Read Txt
        df_1 = df_1.append([[data_1, 0]],ignore_index= True)
df_1.columns = ['Message','label']
#Concat df_0 and df_1
df = pd.concat([df_0, df_1], axis= 0)
df.to_csv('Train_movie.csv')

#Testset
path_2 = 'C:\\Python Software\\Emotional Analysis\\movie\\test\\pos' #Positive test
files_2= os.listdir(path_2) #Read the following txt documents in this file
df_2 = pd.DataFrame()
for b in files_2: 
    position_2 = path_2+'\\'+ b        
    with open(position_2, "r",encoding='utf-8') as f_2:
        data_2 = f_2.read()   #Read Txt
        df_2 = df_2.append([[data_2, 1]],ignore_index= True)
df_2.columns = ['Message','label']
#Negative
path_3 = 'C:\\Python Software\\Emotional Analysis\\movie\\test\\neg'
files_3= os.listdir(path_3) #Read the following txt documents in this file
df_3 = pd.DataFrame()
for c in files_3: 
    position_3 = path_3+'\\'+ c        
    with open(position_3, "r",encoding='utf-8') as f_3:
        data_3 = f_3.read()   #Read Txt
        df_3 = df_3.append([[data_3, 0]],ignore_index= True)
df_3.columns = ['Message','label']
#Concat
df_4 = pd.concat([df_2, df_3], axis= 0)
df_4.to_csv('test_movie.csv')
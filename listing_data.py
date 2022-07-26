import pandas as pd
import os

FolderOutput="/Users/ACER/Work/ML_kominfo/final_project/"
FolderAlphabet="/Users/ACER/Work/ML_kominfo/final_project/alphabet_bisindo/"

dfDataOutput=pd.DataFrame()


FoldersAbjad=os.listdir(FolderAlphabet)
i=0
for FolderAbjad in FoldersAbjad:
    for fileAbjad in os.listdir(os.path.join(FolderAlphabet,FolderAbjad)):
        df=pd.DataFrame([[fileAbjad,FolderAbjad]], columns=['gambar', 'abjad'])
        dfDataOutput=pd.concat([dfDataOutput,df])


dfDataOutput.to_csv(os.path.join(FolderOutput,"data_label.csv"))
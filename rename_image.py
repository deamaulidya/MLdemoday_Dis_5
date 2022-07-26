
import os


FolderAlphabet="/Users/ACER/Work/ML_kominfo/final_project/alphabet_bisindo/"

FoldersAbjad=os.listdir(FolderAlphabet)
for FolderAbjad in FoldersAbjad:
    i=0
    for fileAbjad in os.listdir(os.path.join(FolderAlphabet,FolderAbjad)):
        os.rename(FolderAlphabet+FolderAbjad+'/'+fileAbjad, FolderAlphabet+FolderAbjad+'/'+FolderAbjad+'_'+str(i)+'.jpg')
        i = i+1
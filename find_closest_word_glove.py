import pandas as pd
import csv
import argparse
import numpy as np

def vec(w, df):
  return df.loc[w].values

def find_closest_words(v, df, n):
  words_matrix = df.values
  diff = words_matrix - vec(v,df)
  delta = np.sum(diff * diff, axis=1)
  minlocs = delta.argsort()[:n]
  closest=[]
  for i in minlocs:
    closest.append(words.iloc[i].name)
  return closest

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dim', action="store", dest="EMBEDDING_DIM", default='100')
parser.add_argument('-c','--compare', action="store", dest="compare", default='0')
parser.add_argument('-w','--word', action="store", dest="getword")
parser.add_argument('-n','--num', action="store", dest="num",type=int, default=5)

args = parser.parse_args()
EMBEDDING_DIM=args.EMBEDDING_DIM
compare=args.compare
getword=args.getword
num=args.num


glove_file1="datasets/glove.6B."+str(EMBEDDING_DIM)+"d.txt"
glove_file2="datasets/glove.6B."+str(EMBEDDING_DIM)+"d.bin.txt"


words = pd.read_csv(glove_file2, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
words.drop(words.columns[len(words.columns)-1], axis=1, inplace=True)

print('As per near lossless binarization: ',find_closest_words(getword, words, num))

if compare=='1':
	words = pd.read_csv(glove_file1, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
	words.drop(words.columns[len(words.columns)-1], axis=1, inplace=True)
	print('As per original embedding: ',find_closest_words(getword, words, num))

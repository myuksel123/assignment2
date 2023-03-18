import pandas as panda
from collections import Counter
#data analysis
theFile = panda.read_csv("./train.csv");

theFile.dropna(inplace = True); #drop any null values

numrows = len(theFile); #gives the rows of data, excluding the title row
print("the number of rows is " + str(numrows));

df_negatives = theFile[theFile['Sentiment']==0];
numNegatives = len(df_negatives);
print("the number of negatives is " + str(numNegatives));

df_positives= theFile[theFile['Sentiment']==1];
numPositives = len(df_positives);
print("the number of positives is " + str(numPositives));


#text preprocessing
#remove any non-letters from the file
for i in theFile['Text']:
    for character in i:
        if not character.isalpha() and not character == ' ':
            print(character+"!!!! ENDING");
#for i in theFile['Text']:
#   if 
#feature extraction

#model

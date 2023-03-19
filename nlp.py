import pandas as panda
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#data analysis
Training = panda.read_csv("./train.csv");
Testing = panda.read_csv("./test.csv");

Training.dropna(inplace = True); #drop any null values
Testing.dropna(inplace = True);

numrows = len(Training); #gives the rows of data in testing, excluding the title row
print("the number of training rows is " + str(numrows));

df_negatives = Training[Training['Sentiment']==0];
numNegatives = len(df_negatives);
print("the number of training negatives is " + str(numNegatives));

df_positives= Training[Training['Sentiment']==1];
numPositives = len(df_positives);
print("the number of training positives is " + str(numPositives));


#text preprocessing
#remove any non-letters and non-spaces  from the file
Training['Text'] = Training['Text'].str.replace('[^a-zA-Z ]','', regex=True);
Testing['Text'] = Testing['Text'].str.replace('[^a-zA-Z ]','', regex=True);

#make the letters all lowercase
Training['Text'] = Training['Text'].str.lower();
Testing['Text'] = Testing['Text'].str.lower();

#tokenize and make a bag of the words
bagOfWords = CountVectorizer(token_pattern=r'\b\w+\b');

trainX = bagOfWords.fit_transform(Training['Text']);
testX = bagOfWords.transform(Testing['Text']);
testY = Testing['Sentiment'];

#model
myModel = LogisticRegression(max_iter = 1100000);
trainY = Training['Sentiment'];
myModel.fit(trainX,trainY);

predicted = myModel.predict(testX);

#showing how accurate the predictions were:
print(classification_report(predicted, testY));
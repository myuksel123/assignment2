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
print("the number of rows is " + str(numrows));

df_negatives = Training[Training['Sentiment']==0];
numNegatives = len(df_negatives);
print("the number of negatives is " + str(numNegatives));

df_positives= Training[Training['Sentiment']==1];
numPositives = len(df_positives);
print("the number of positives is " + str(numPositives));


#text preprocessing
#remove any non-letters and non-spaces  from the file
Training['Text'] = Training['Text'].str.replace('[^a-zA-Z ]','', regex=True);
Testing['Text'] = Testing['Text'].str.replace('[^a-zA-Z ]','', regex=True);

#make the letters all lowercase
Training['Text'] = Training['Text'].str.lower();
Testing['Text'] = Testing['Text'].str.lower();

#tokenize the words

#feature extraction
#wordfreq = {}
#for sentence in theFile['Text']:
#    tokens = nltk.word_tokenize(sentence)
#    for token in tokens:
#        if token not in wordfreq.keys():
#            wordfreq[token] = 1
#        else:
#            wordfreq[token] += 1
#bagOfWords = heapq.nlargest(250,wordfreq,key= wordfreq.get);
#bagOfwords = heapq.nsmallest(200,bagOfWords,key= bagOfWords.get);
#the reason I got the top 250 and then the bottom 200 of that 250
#is because I wanted to exclude the stop words such as "I" that 
#would be in the top 50, but include the words that are used
#a little less often than those, such as "like" or "love"

# we will use the CounterVectorizer to create a bag of words
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
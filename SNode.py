import twitter
import re
import io
import csv
import time
import nltk
import pickle
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords

class SNode:


    def __init__(self):
        self.stopwords_list = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])

    def buildTrainingSetFromFile(self, tweetDataFile):
        trainingDataSet = []
        file = io.open(tweetDataFile, 'r', encoding="utf-8")
        line = file.readline()

        while line:
            trainingDataSet.append({"text":line[2:], "label":line[0]})
            line = file.readline()

        file.close()
        return trainingDataSet

    def processTweet(self, tweet):
        tweet = tweet.lower() # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
        tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
        return [word for word in tweet if word not in self.stopwords_list]

    def preProcessTweets(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append((self.processTweet(tweet["text"]),tweet["label"]))
        return processedTweets

    def buildVocabulary(self, trainingDataSet):
        all_words = []

        for (words, sentiment) in trainingDataSet:
            all_words.extend(words)

        wordlist = nltk.FreqDist(all_words)
        word_features = wordlist.keys()

        return word_features

    def extract_features(self, tweet):
        tweet_words = set(tweet)
        features = {}
        for word in self.word_features:
            features['contains(%s)' % word] = (word in tweet_words)
        return features

    def buildTestSet(self, search_keyword, tweetCount, start, end):
        try:
            tweets_fetched = self.twitter_api.GetSearch(search_keyword, count = tweetCount, since=start, until=end)

            # print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)
            return [{"text":status.text, "label":None} for status in tweets_fetched]
        except:
            print("Unfortunately, something went wrong..")
            return None

    def train(self, trainFile):
        trainingDataSet = self.buildTrainingSetFromFile(trainFile)
        processedTrainingSet = self.preProcessTweets(trainingDataSet)
        self.word_features = self.buildVocabulary(processedTrainingSet)
        trainingFeatures = nltk.classify.apply_features(self.extract_features, processedTrainingSet)
        self.NBayesClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)

    def load(self, pickleFile):
        data = pickle.load(open(pickleFile, "rb"))
        self.NBayesClassifier = data[0]
        self.word_features = data[1]
        self.NBayesClassifier.show_most_informative_features(15)

    def save(self, pickleFile):
        data = [self.NBayesClassifier, list(self.word_features)]
        pickle.dump(data, open(pickleFile, "wb"))

    def predict(self, search_terms, tweetCount, start, end):
        #INSERT TWITTER CREDENTIALS HERE
        self.twitter_api = twitter.Api(consumer_key='',
                                consumer_secret='',
                                access_token_key='',
                                access_token_secret='')
        testDataSet = []
        for term in search_terms:
            testDataSet.extend(self.buildTestSet(term, tweetCount, start, end))
        processedTestSet = self.preProcessTweets(testDataSet)
        NBResultLabels = [self.NBayesClassifier.classify(self.extract_features(tweet[0])) for tweet in processedTestSet]

        if NBResultLabels.count('1') > NBResultLabels.count('0'):
            # print("Overall Positive Sentiment")
            sentiment_score = NBResultLabels.count('1')/len(NBResultLabels)
            # print("Positive Sentiment Score = " + str(sentiment_score))
        else:
            # print("Overall Negative Sentiment")
            sentiment_score =-1*( NBResultLabels.count('1')/len(NBResultLabels))
            # print("Negative Sentiment Score = " + str(sentiment_score))
        return sentiment_score

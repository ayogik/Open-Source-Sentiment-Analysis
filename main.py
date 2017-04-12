from __future__ import division
import tweepy
from operator import itemgetter
from tweepy import OAuthHandler
from collections import Counter
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import math
import os.path
import numpy as np
from watson_developer_cloud import AlchemyLanguageV1

try:
    import json
except ImportError:
    import simplejson as json;

positivewords = []
positivewordsprefound = []
positivewordscounttimes = []
pos = "positive"
with open(pos) as f:
    for a in f:
        b = a.split('|')
        positivewords.append(b[0])
        positivewordsprefound.append(float(b[1]))
        positivewordscounttimes.append(int(b[2]))

negativewords = []
negativewordsprefound = []
negativewordscounttimes = []
neg = "negative"
with open(neg) as f:
    for a in f:
        b = a.split('|')
        negativewords.append(b[0])
        negativewordsprefound.append(float(b[1]))
        negativewordscounttimes.append(int(b[2]))

consumer_key = 'key'
consumer_secret = 'secret'
access_token = 'token'
access_secret = 'secret'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

stopwords = stopwords.words('english') + list(string.punctuation)

myfile = "tweets"
if not os.path.isfile(myfile):
    print "reading from twitter"
    num_of_tweets = 10
    target = open(myfile, 'w')
    for tweet in tweepy.Cursor(api.user_timeline).items(num_of_tweets):
        target.write(" |")
        target.write(tweet._json["text"])
        target.write("\n")
    target.close()

num_of_tweets = 0
tweetlists = []
originaltweets = []

with open(myfile) as f:
    try:
        for a in f:
            b = a.split('|')
            intendedsentiment = b[0]
            line = b[1]
            originaltweets.append([line, intendedsentiment])
            sentence = line.split()
            sentence = [s.encode('utf-8') for s in sentence]
            sentence = [s.lower() for s in sentence]
            sentence = [s.translate(None, '!@#?._,') for s in sentence]
            sentencenostop = [s for s in sentence if s not in stopwords]
            tweetlists.append(sentencenostop)
            num_of_tweets += 1
    except:
        print "something's wrong"

tweetlists_frequency = []
for n in range(0,num_of_tweets):
    list_at = tweetlists[n]
    wordcount = Counter(list_at)
    tweetlists_frequency.append(wordcount)

wordlists_freq = dict(sum(tweetlists_frequency, Counter()))
wordlists = wordlists_freq.keys()

wordnumtweets = defaultdict(int)
for word in wordlists:
    for tweet in tweetlists:
        if word in tweet:
            wordnumtweets[word] += 1

pairnumtweets = defaultdict(lambda: defaultdict(int))
for tweet in tweetlists:
    for a in range(0, len(tweet)-1):
        for b in range(a+1, len(tweet)):
            word1, word2 = sorted([tweet[a], tweet[b]])
            if word1 != word2:
                pairnumtweets[word1][word2] += 1

wordprob = defaultdict(float)
for a in wordnumtweets.keys():
    wordprob[a] = (wordnumtweets[a]/num_of_tweets)

pairprob = defaultdict(lambda: defaultdict(int))
for a in pairnumtweets.keys():
    for b in pairnumtweets[a].keys():
        pairprob[a][b] = pairnumtweets[a][b]/num_of_tweets


def pmi(t1, t2):
    w1, w2 = sorted([t1,t2])
    x = pairprob[w1][w2]/(wordprob[w1]*wordprob[w2])
    if x == 0:
        return 0
    return math.log(x, 2)


def search(word, lis):
    lower_word = word.lower()
    to_return = 0
    for i in range(1, len(lis)):
        if lis[i] == lower_word:
            to_return = i
            break
        else:
            to_return = 0
    return to_return


def semantic_orientation(t):
    pos_orientation = 0.0
    neg_orientation = 0.0
    for t1 in positivewords:
        if wordprob[t1] != 0.0:
            pos_orientation += pmi(t, t1)
    for t2 in negativewords:
        if wordprob[t2] != 0.0:
            neg_orientation += pmi(t, t2)
    term_so = pos_orientation - neg_orientation

    index = search(t, positivewords)
    neg = False
    if index == 0:
        index = search(t, negativewords)
        neg = True

    if neg and index != 0:
        s_o = (term_so + negativewordsprefound[index])/2
        negativewordsprefound[index] = (negativewordsprefound[index]*negativewordscounttimes[index] + term_so) /\
                                       (negativewordscounttimes[index]+1)
        negativewordscounttimes[index] += 1

    elif not neg and index != 0:
        s_o = (term_so + positivewordsprefound[index])/2
        positivewordsprefound[index] = (positivewordsprefound[index]*positivewordscounttimes[index] + term_so) /\
                                       (positivewordscounttimes[index]+1)
        positivewordscounttimes[index] += 1

    else:
        s_o = term_so
        if term_so >= 0:
            positivewords.append(t.lower())
            positivewordsprefound.append(term_so)
            positivewordscounttimes.append(1)
        elif term_so < 0:
            negativewords.append(t.lower())
            negativewordsprefound.append(term_so)
            negativewordscounttimes.append(1)

    return s_o

alchemyapikey = "key"
alchemy_language = AlchemyLanguageV1(api_key=alchemyapikey)

final_tweet_orientations = defaultdict(int)
for a in range(0, len(tweetlists)):
    tweet_orientation = 0.0
    for word in tweetlists[a]:
        tweet_orientation += semantic_orientation(word)

    ts = ""
    if tweet_orientation > 0:
        ts += "positive"
    else:
        ts += "negative"

    originaltweets[a].append(ts)
'''
    alchemy_sentiment = ""

    retval = alchemy_language.sentiment(text=originaltweets[a][0])
    alchemy_sentiment += retval["docSentiment"]["type"]

    originaltweets[a].append(alchemy_sentiment)
'''
newpos = []
newposval = []
newposcount = []

newneg = []
newnegval = []
newnegcount = []

for i in range(0, len(positivewords)):
    if positivewordsprefound[i] < 0:
        newneg.append([positivewords[i], positivewordsprefound[i], positivewordscounttimes[i]])
    else:
        newpos.append([positivewords[i], positivewordsprefound[i], positivewordscounttimes[i]])

for i in range(0, len(negativewords)):
    if negativewordsprefound[i] > 0:
        newpos.append([negativewords[i], negativewordsprefound[i], negativewordscounttimes[i]])
    else:
        newneg.append([negativewords[i], negativewordsprefound[i], negativewordscounttimes[i]])

editpositive = "positive"
with open(editpositive, 'w') : pass
update1 = open(editpositive, 'w')
for a in newpos:
    update1.write(a[0])
    update1.write('|')
    update1.write(str(a[1]))
    update1.write('|')
    update1.write(str(a[2]))
    update1.write("\n")
update1.close()

editnegative = "negative"
with open(editnegative, 'w') : pass
update2 = open(editnegative, 'w')
for a in newneg:
    update2.write(a[0])
    update2.write('|')
    update2.write(str(a[1]))
    update2.write('|')
    update2.write(str(a[2]))
    update2.write("\n")
update2.close()

outputfile = "outputfile.csv"
with open(outputfile, 'w') : pass
goingout = open(outputfile, 'w')
for a in range(0,len(originaltweets)):
    goingout.write(str(a+1))
    goingout.write(',')
    goingout.write(originaltweets[a][1])
    goingout.write(',')
    goingout.write(originaltweets[a][2])
    #goingout.write(',')
    #goingout.write(originaltweets[a][3])
    goingout.write("\n")
goingout.close()

'''
finalsentimentcounter = 0
for a in originaltweets:
    if a[2] == "negative":
        finalsentimentcounter -= 1
    elif a[2] == "positive":
        finalsentimentcounter += 1

if finalsentimentcounter<0:
    status = api.update_status(status="Cheer up!")
'''
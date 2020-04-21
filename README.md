# Twitter Competition Net
Twitter Competition Net is a model which uses specific tweets to gauge public opinion towards a handful of
tickers. These tickers represent an environment of competition, meaning the tickers are of companies
in the same market and who are direct competitors to the target.

# Set Up
Please retrieve your tiwtter api keys from the developer site and input them in `SNode.py`

```
self.twitter_api = twitter.Api(consumer_key='INSERT_KEY_HERE',
                        consumer_secret='INSERT_KEY_HERE',
                        access_token_key='INSERT_KEY_HERE',
                        access_token_secret='INSERT_KEY_HERE')
```

Running `python TCN.py` will construct the training set and test set, train the model and test it.

# Imports
#!pip install streamlit -q <- co-lab
#from this import s
#from turtle import title
import pandas as pd
import numpy as np
import yfinance as yf
import hvplot.pandas
import seaborn as sn
import streamlit as st
import holoviews as hv
import nltk as nltk
import tweepy 
import re
import string
import warnings
import matplotlib.pyplot as plt

#from st_aggrid import AgGrid
from finta import TA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.porter import PorterStemmer
from tweepy import OAuthHandler 
from tweepy import Cursor 




# Setup environment variable
model  = SVC()
scaler = MinMaxScaler()
analyzer = SentimentIntensityAnalyzer()
stemmer = PorterStemmer()
warnings.filterwarnings('ignore')
#nltk.download('vader_lexicon')
hv.extension('bokeh')

# Twitter API information
api_key='2LEkmltQnjgt4NEwAiU3wKfiU'
api_key_secret='Jg0Fl7wkEAVSbJnXpHXXZEySZka9Dg3FxW8H3tqCfaoFjrsowc'
acc_token='143141229-1gfi2d61Aco7beRwqgxTo8DdzjWmaKLmqPZgVepA'
acc_secret='4Cxbo1eNgS6DF69gF7vJ01IXE2hhU7Uo3qAjgvskhTTfo'



# Function to Retrieve ticker data
def ticker_data_history(ticker, column_drop, rename_column, per = 'max', int_period = '1wk'):
    x = yf.Ticker(ticker)                                     # Get ticker data
    x_data = x.history(period=per,interval = int_period)      # Extract historical data 
    x_data = x_data.dropna()                                  # Remove any nas
    if column_drop != "":                                     # If there are any columns to drop
        x_data = x_data.drop(columns=column_drop)             # drop those columns
    if rename_column != "":                                   # Rename 'Close' to given name
        x_data = x_data.rename(columns = {'Close':rename_column}) 

    return x_data   

# Function to Run analysis
def run_correlation(sel):
    drop_columns = ['Dividends','Stock Splits' ,'Low','High','Open','Volume']
    tickers = []
    tickers.append('SPY')
    tickers.append(sel) 
    period   ='2y'
    interval ='1h'
    df_data = pd.DataFrame()
    for ticker in tickers:
        df_data[ticker] = ticker_data_history(ticker, drop_columns, ticker, period)
    df_data.dropna(inplace=True)  # Drop any nas

    # Correlation calculation. Retrieve all sector data
    tickers  = ['SPY', 'XLK', 'XLV', 'XLI', 'XLP', 'XLE', 'XLY', 'XLB']
    df_data1 = pd.DataFrame()    
    for ticker in tickers:
        df_data1[ticker] = ticker_data_history(ticker, drop_columns, ticker)
    df_data1.dropna(inplace=True)  # Drop any nas
    correlation = df_data1.corr()

    # Plot heat map for all sector data
    sn.set_theme(style="darkgrid")
    fig, ax = plt.subplots()
    correlation_map = sn.heatmap(correlation,vmin = 0.0,vmax=1.0)
    
    return df_data, correlation, fig


# Function to Run analysis
def run_model(sel, name):
    # Get TRX Gold Corp data from yahoo finance
    drop_columns = ['Dividends','Stock Splits']
    period   ='2y'
    interval ='1h'
    ticker   = '^TNX'
    tnx = ticker_data_history(ticker, drop_columns, "", period, interval)
    
    ticker = sel
    sel_tick = ticker_data_history(ticker, drop_columns, "", period, interval)

    # Make the both dataframe same length
    if len(tnx) > len(sel_tick):
        tnx = tnx.iloc[(len(tnx) - len(sel_tick)):]
    elif len(tnx) < len(sel_tick):
        sel_tick = sel_tick.iloc[len(sel_tick) - len(tnx):]
        
    # Calculate Exponential Weighted Moving Average using Finta lib and
    # Set buy_signal
    ema = pd.DataFrame(TA.EMA(sel_tick,9), index = sel_tick.index)   # EMA with 9 periods
    ema = ema.rename(columns = {'9 period EMA' : 'ema_5'})      # Rename the column
    ema['ema_21'] = TA.EMA(sel_tick,21)                         # EMA with 21 periods
    ema['signal'] = 0.0                                         # Set column name = 'signal'
    ema['signal'][9:]=np.where(ema['ema_5'][9:]>ema['ema_21'][9:],1.0,0.0) # calculate signal


    # Create sector data frame
    col = ["Close", "tnx", "ema_5", "ema_21", "signal", "entry/exit"]
    sector_data = pd.DataFrame(columns = col, index=ema.index)
    sector_data['Close']  = sel_tick['Close'].values                 # Add xlk Close price column
    sector_data['tnx']    = tnx['Close'].values                      # Add xlk Close tnx column
    sector_data['ema_5']  = ema['ema_5'].values                      # Add ema_5 column
    sector_data['ema_21'] = ema['ema_21'].values                     # Add ema_21 column
    sector_data['signal'] = ema['signal'].values                     # Add signal column
    sector_data['entry/exit'] = ema['signal'].diff()                 # Calculate and Add diff on signal
    sector_data.dropna(inplace=True)                                 # Drop any nas and assign to new DF=sector_data

    # RSI Calculation
    sel_rsi = pd.DataFrame(TA.RSI(sel_tick))                    # Calculate RSI value
    sel_rsi = sel_rsi.dropna()                                  # Drop any nas
    sel_rsi = sel_rsi[sel_rsi['14 period RSI'] != 100.000000]   # drop any rows where '14 period RSI' = 100.000000

    try:
        sector_data.insert(2,'rsi_14',sel_rsi['14 period RSI']) # Insert rsi to df=sector_data
    except:
        sector_data['rsi_14'] = sel_rsi['14 period RSI']        # Add rsi to df=sector_data

    sector_data.dropna(inplace=True)                            # Drop any nas
    X_feature = sector_data.iloc[:,:4]                          # Extract first 4 columns for X-Feature

    # Select y value and setup training/testing data
    y = sector_data.iloc[:,-2].values           # Select column = 'signal' as y value
    split   = int(0.75*len(sel_tick))           # Split 75% of data based on xlk's total data
    X_train = X_feature[:split]                 # Assign 75% to X_train
    X_test  = X_feature[split:]                 # Assign 25% to X_test
    y_train = y[:split]                         # Assign 75% to y_train
    y_test  = y[split:]                         # Assign 25% to y_train

    # Shape, fit, scale  and train data
    t = X_train.shape                           # Shape the X_train data
    t = y_test.shape                            # Shape the y_test data
    x_train_scaled = scaler.fit(X_train)        # Fit the X_train data using MinMaxScaler
    x_train_scaled = scaler.transform(X_train)  # Now, scale the feature training
    x_test_scaled  = scaler.transform(X_test)   # Now, scale the feature testing
    model.fit(x_train_scaled, y_train)          # Train the model using sklearn.svm
    predictions = model.predict(x_test_scaled)  # Get the predition from the model

    # Print classification report
    report_test = classification_report(y_test,predictions)
    report_test1 = classification_report(y_test,predictions, output_dict=True, digits=4)
    report_df = pd.DataFrame(report_test1)
    report_df.drop(columns=['weighted avg', 'accuracy', 'macro avg'], inplace=True)
    report_df = report_df.iloc[0:3]


    col1,col2 = st.columns([2,2])
    y_test = pd.DataFrame(y_test,columns =['y_t'])                # Convert y_test and name the column  

    predictions = pd.DataFrame(predictions,columns = ['pred'])    # Create df of preditions
    predictions['y_pred'] = predictions['pred'].diff()            # Calculate diff() based on 'pred'
    predictions['y_actual'] = y_test['y_t'].diff()                # Add & calculate y_actual
    predictions.dropna(inplace=True)                              # Drop all nas
    predictions['close'] = X_feature.iloc[split+1:,0].values      # Add close price from X_feature

    t = pd.DataFrame(predictions['y_pred'])
    if t['y_pred'].iat[-1] == 1.0:
        decission = f"We Recommend a Buy for the {name} Sector"
    elif t['y_pred'].iat[-1] == -1.0:
        decission = f"We Recommend a Sell for the {name} Sector"
    else:
        decission = f"We Recommend to stay in cash for the {name} Sector at this time"

    #Total buys
    days_bought=predictions[predictions['y_actual']== -1.0]

    #Plot overall map
    entry = predictions[predictions['y_pred'] == 1.0]['close'].hvplot.scatter(
        color='purple',
        marker='^',
        legend=False,
        ylabel='Price in $',
        width=1000,
        height=400)

    # Visualize exit positions relative to close price
    exit = predictions[predictions['y_pred'] == -1.0]['close'].hvplot.scatter(
        color='orange',
        marker='v',
        legend=False,
        ylabel='Price in $',
        width=1000,
        height=400)

    # Visualize the close price for the investment
    security_close = predictions[['close']].hvplot(
        line_color='lightgray',
        ylabel='Price in $',
        width=1000,
        height=400)


    # Overlay the plots
    entry_exit_plot = security_close *  entry * exit
    entry_exit_plot.opts(
        title="Sector Buy/Sell Signals"
    )

    #Graph the classification report
    report_df = report_df.iloc[0:3]
    bar = report_df.hvplot.bar(title='Performance Metrics',  xlabel='Metrics', ylabel='Classification')

    return ema, sector_data, X_feature, report_test, y_test, predictions, days_bought, entry_exit_plot, bar, decission

# Sentiment calculation from twitter
def get_twitter_auth():
    """
    @return:
        - the authentification to Twitter
    """
    try:
        consumer_key = api_key
        consumer_secret = api_key_secret
        access_token = acc_token
        access_secret = acc_secret
        
    except KeyError:
        st.write("Twitter Environment Variable not Set\n")
        return 0
        
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    
    return auth

def get_twitter_client():
    """
    @return:
        - the client to access the authentification API
    """
    auth = get_twitter_auth()
    client = tweepy.API(auth, wait_on_rate_limit=True)
    return client

def get_tweets_from_user(twitter_user_name, page_limit=16, count_tweet=20):
    """
    @params:
        - twitter_user_name: the twitter username of a user (company, etc.)
        - page_limit: the total number of pages (max=16)
        - count_tweet: maximum number to be retrieved from a page
        
    @return
        - all the tweets from the user twitter_user_name
    """
    client = get_twitter_client()
    
    all_tweets = []
    
    for page in Cursor(client.user_timeline, 
                        screen_name=twitter_user_name, 
                        count=count_tweet).pages(page_limit):
        for tweet in page:
            parsed_tweet = {}
            parsed_tweet['date'] = tweet.created_at
            parsed_tweet['author'] = tweet.user.name
            parsed_tweet['twitter_name'] = tweet.user.screen_name
            parsed_tweet['text'] = tweet.text
            parsed_tweet['number_of_likes'] = tweet.favorite_count
            parsed_tweet['number_of_retweets'] = tweet.retweet_count
                
            all_tweets.append(parsed_tweet)
    
    # Create dataframe 
    df = pd.DataFrame(all_tweets)
    
    # Revome duplicates if there are any
    df = df.drop_duplicates( "text" , keep='first')
    
    return df

# Sentiment calculation based on compound score
def get_sentiment(score):
    """
    Calculates the sentiment based on the compound score.
    """
    result = 0  # Neutral by default
    if score >= 0.05:  # Positive
        result = 1
    elif score <= -0.05:  # Negative
        result = -1

    print(f"Compound: {score}, Result: {result}")
    
    return result
    
#function for remove pattern in the input text
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt

def sentiment_analysis():
    # making a dataframe with the bloomberg news
    df1 = get_tweets_from_user("markets")
    print("Data Shape: {}".format(df1.shape))
    df1.columns = df1.columns.str.replace('text', 'tweet')
    bloomberg = df1['tweet']

    # making a dataframe with the wall street journal news
    df2 = get_tweets_from_user("wsj")
    print("Data Shape: {}".format(df2.shape))
    df2.columns = df2.columns.str.replace('text', 'tweet')
    wall_street = df2['tweet']

    # making a dataframe with the yahoo finance news
    df3 = get_tweets_from_user("yfinanceplus")
    print("Data Shape: {}".format(df3.shape))
    df3.columns = df3.columns.str.replace('text', 'tweet')
    yahoo_finance = df3['tweet']

    #Concat all the tweets from the news accounts
    a = {'Bloomberg':bloomberg.values, 'Wall Street':wall_street.values, 'Yahoo Finance':yahoo_finance}
    all_tweet = pd.DataFrame.from_dict(a, orient='index')
    all_tweet = all_tweet.transpose()
    #st.dataframe(all_tweet)                       #Display data

    # #changing the name all_tweet to df again
    # df = all_tweet

    # #combine all tweets per news in one column
    # df["tweet"] = df["Bloomberg"].astype(str) + df["Wall Street"].astype(str) + df["Yahoo Finance"].astype(str)

    # #remove twitter users
    # df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")

    # #remove special characters, numbers and punctuations
    # df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")

    # #remove short words
    # df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
    # df.head()

    # tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())

    # #stem the words
    # tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])

    # #combine words into single sentence
    # for i in range(len(tokenized_tweet)):
    #     tokenized_tweet[i] = " ".join(tokenized_tweet[i])

    # df['clean_tweet'] = tokenized_tweet
    # df = df.drop(['tweet'], axis=1)

    # # Sentiment scores dictionaries
    # tweet_sent = {
    #     "tweet_compound": [],
    #     "tweet_pos": [],
    #     "tweet_neu": [],
    #     "tweet_neg": [],
    #     "tweet_sent": [],
    # }

    # # Get sentiment for the text and the title
    # for index, row in df.iterrows():
    #     try:
    #         # Sentiment scoring with VADER
    #         tweet_sentiment = analyzer.polarity_scores(row["clean_tweet"])
    #         tweet_sent["tweet_compound"].append(tweet_sentiment["compound"])
    #         tweet_sent["tweet_pos"].append(tweet_sentiment["pos"])
    #         tweet_sent["tweet_neu"].append(tweet_sentiment["neu"])
    #         tweet_sent["tweet_neg"].append(tweet_sentiment["neg"])
    #         tweet_sent["tweet_sent"].append(get_sentiment(tweet_sentiment["compound"]))
    #     except AttributeError:
    #         pass

    # # Attaching sentiment columns to the News DataFrame
    # tweet_sentiment_df = pd.DataFrame(tweet_sent)
    # st.dataframe(tweet_sentiment_df)

    # # Describe dataframe
    # st.dataframe(tweet_sentiment_df.describe())

    # making a dataframe with the CMTAssociation news
    df4 = get_tweets_from_user("CMTAssociation")
    print("Data Shape: {}".format(df4.shape))
    df4.columns = df4.columns.str.replace('text', 'tweet')
    CMTAssociation = df4['tweet']

    # making a dataframe with the ParetsJc news
    df5 = get_tweets_from_user("ParetsJc")
    print("Data Shape: {}".format(df5.shape))
    df5.columns = df5.columns.str.replace('text', 'tweet')
    ParetsJc = df5['tweet']

    # making a dataframe with the allstarcharts news
    df6 = get_tweets_from_user("allstarcharts")
    print("Data Shape: {}".format(df6.shape))
    df6.columns = df6.columns.str.replace('text', 'tweet')
    allstarcharts = df6['tweet']

    #Concat all the tweets from the news accounts
    a = {'CMT Association':CMTAssociation.values, 'RSI Wizard':ParetsJc.values, 'J.C. Parets':allstarcharts.values}
    all_people_tweet = pd.DataFrame.from_dict(a, orient='index')
    all_people_tweet = all_people_tweet.transpose()
    new_df = all_people_tweet
    #st.dataframe(all_people_tweet)  

    # #combine all tweets per news in one column
    # new_df["tweet"] = new_df["CMT Association"].astype(str) + new_df["RSI Wizard"].astype(str) + new_df["J.C. Parets"].astype(str)

    # #remove twitter users
    # new_df['clean_tweet'] = np.vectorize(remove_pattern)(new_df['tweet'], "@[\w]*")

    # #remove special characters, numbers and punctuations
    # new_df['clean_tweet'] = new_df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")

    # #remove short words
    # new_df['clean_tweet'] = new_df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))

    # tokenized = new_df['clean_tweet'].apply(lambda x: x.split())

    # #stem the words
    # #from nltk.stem.porter import PorterStemmer
    # #stemmer = PorterStemmer()

    # tokenized = tokenized.apply(lambda sentence: [stemmer.stem(word) for word in sentence])

    # #combine words into single sentence
    # for i in range(len(tokenized)):
    #     tokenized[i] = " ".join(tokenized[i])

    # new_df['clean_tweet'] = tokenized
    # new_df = new_df.drop(['tweet'], axis=1)

    # #from nltk.sentiment.vader import SentimentIntensityAnalyzer
    # #analyzer = SentimentIntensityAnalyzer()

    # # Sentiment scores dictionaries
    # tweet_sent2 = {
    #     "tweet_compound2": [],
    #     "tweet_pos2": [],
    #     "tweet_neu2": [],
    #     "tweet_neg2": [],
    #     "tweet_sent2": [],
    # }

    # # Get sentiment for the text and the title
    # for index, row in new_df.iterrows():
    #     try:
    #         # Sentiment scoring with VADER
    #         tweet_sentiment2 = analyzer.polarity_scores(row["clean_tweet"])
    #         tweet_sent2["tweet_compound2"].append(tweet_sentiment2["compound"])
    #         tweet_sent2["tweet_pos2"].append(tweet_sentiment2["pos"])
    #         tweet_sent2["tweet_neu2"].append(tweet_sentiment2["neu"])
    #         tweet_sent2["tweet_neg2"].append(tweet_sentiment2["neg"])
    #         tweet_sent2["tweet_sent2"].append(get_sentiment(tweet_sentiment2["compound"]))
    #     except AttributeError:
    #         pass

    # # Attaching sentiment columns to the News DataFrame
    # tweet_sentiment_df2 = pd.DataFrame(tweet_sent2)

    # st.dataframe(tweet_sentiment_df2)

    # # Describe dataframe
    # st.dataframe(tweet_sentiment_df2.describe())


    return all_tweet, all_people_tweet



def main():

    #tickers = ['SPY', 'XLK', 'XLV', 'XLI', 'XLP', 'XLE', 'XLY', 'XLB']
    st.header("Sector Investing Strategy")
    select_value = ["Technology", "Health Care", "Industrial", "Consumer Staples", "Energy", "Consumer Discretionary", "iShares Core Canadian EFT"]
    sector = st.selectbox("Select the sector", select_value)

    if st.button("Submit"):
        st.caption(f"Selection is {sector}. Collecting data for selected sector")

        if sector == 'Technology':
            tick = 'XLK'
        elif sector == 'Health Care':
            tick = 'XLV'
        elif sector == 'Industrial':
            tick = 'XLI'
        elif sector == 'Consumer Staples':
            tick = 'XLP'
        elif sector == 'Energy':
            tick = 'XLE'
        elif sector == 'Consumer Discretionary':
            tick = 'XLY'
        else: 
            tick = 'XLB'

        df_data, correlation, fig = run_correlation(tick)
        ema, sector, feature, report, y, predictions, bought, entry_plot, bar, decission = run_model(tick, sector)
        all_tweet, all_people_tweet = sentiment_analysis()

        col1,col2,col3 = st.columns([3,1,3])
        
        with col1:
            st.write(f"Ticker data and Correlation Data") 
            st.dataframe(df_data)

        with col2:
            st.write(f"Correlation Data")
            st.dataframe(correlation)

        with col3:
            st.write(f"y_test data")
            st.dataframe(y)
            
        col1,col2 = st.columns([2,2])

        st.write(f"Moving Average Data - 5 and 21 windows")
        st.dataframe(ema)

        st.write(f"Combined sector data")
        st.dataframe(sector)
        
        st.write(f"Relative Strength Index")
        st.dataframe(feature)
        
        st.write(f"Classification Report")
        st.text(report)

        # with col1:
        #     st.write(f"Predicted data")
        #     st.dataframe(predictions)

        # with col2:            
        #     st.write(f"Number of Days bought")
        #     st.dataframe(bought)
        
        st.write("\n")
        st.subheader(f"Heatmap for all sector")
        st.write(fig, use_container_width=True)

        st.write("\n")
        st.subheader(f"Model performance plot")
        st.write(hv.render(entry_plot, backend='bokeh'))

        st.write("\n")
        st.subheader(f"Performance Metrics")
        st.write(hv.render(bar, backend='bokeh'))

        st.write("\n")
        st.header(f"Sentiment Analysis")
        st.subheader(f"News Sentiment Tweets:")
        pd.set_option('display.max_columns', None)

        st.dataframe(all_tweet)
        #AgGrid(all_tweet)
        st.subheader(f"Traders Sentiment Tweets: ")
        st.dataframe(all_people_tweet)
        
        st.header(decission)



if __name__ == '__main__':
    main()




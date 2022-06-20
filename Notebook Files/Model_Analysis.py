# Imports
#!pip install streamlit -q <- co-lab
from this import s
from turtle import title
import pandas as pd
import numpy as np
import yfinance as yf
import hvplot.pandas
import seaborn as sn
import streamlit as st
import holoviews as hv

from finta import TA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# Setup environment variable
model  = SVC()
scaler = MinMaxScaler()
hv.extension('bokeh')

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
    df_data = pd.DataFrame()
    for ticker in tickers:
        #st.text(f"{ticker}")
        df_data[ticker] = ticker_data_history(ticker, drop_columns, ticker)
    df_data.dropna(inplace=True)  # Drop any nas

    col1,col2,col3 = st.columns([3,1,3])
    with col1:
        st.write(f"Ticker data and Correlation Data")
        st.dataframe(df_data)

    with col2:
        correlation = df_data.corr()
        st.write(f"Correlation Data")
        st.dataframe(correlation)

    with col3:    
        st.write(f"Correlation HeatMap")
        sn.set_theme(style="darkgrid")
        fig, ax = plt.subplots()
        correlation_map = sn.heatmap(correlation,vmin = 0.0,vmax=1.0)
        #figure = correlation_map.get_figure()    
        #figure.savefig('correlation_map.png', dpi=100)
        st.write(fig, use_container_width=True)
    
    return

# Function to Run analysis
def run_model(sel):
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
    st.write(f"Moving Average Data - 5 and 21 windows")
    st.dataframe(ema)

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
    st.write(f"Combined sector data")
    st.dataframe(sector_data)

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
    #sector_data
    st.dataframe(X_feature)


    # Select y value and setup training/testing data
    y = sector_data.iloc[:,-2].values           # Select column = 'signal' as y value
    split   = int(0.75*len(sel_tick))           # Split 75% of data based on xlk's total data
    X_train = X_feature[:split]                 # Assign 75% to X_train
    X_test  = X_feature[split:]                 # Assign 25% to X_test
    y_train = y[:split]                         # Assign 75% to y_train
    y_test  = y[split:]                         # Assign 25% to y_train

    # Shape, fit, scale  and train data
    X_train.shape                               # Shape the X_train data
    y_test.shape                                # Shape the y_test data
    x_train_scaled = scaler.fit(X_train)        # Fit the X_train data using MinMaxScaler
    x_train_scaled = scaler.transform(X_train)  # Now, scale the feature training
    x_test_scaled  = scaler.transform(X_test)   # Now, scale the feature testing
    model.fit(x_train_scaled, y_train)          # Train the model using sklearn.svm
    predictions = model.predict(x_test_scaled)  # Get the predition from the model

    # Print classification report
    report_test = classification_report(y_test,predictions)
    st.text(report_test)

    col1,col2 = st.columns([2,2])
    y_test = pd.DataFrame(y_test,columns =['y_t'])                # Convert y_test and name the column
    

    predictions = pd.DataFrame(predictions,columns = ['pred'])    # Create df of preditions
    predictions['y_pred'] = predictions['pred'].diff()            # Calculate diff() based on 'pred'
    predictions['y_actual'] = y_test['y_t'].diff()                # Add & calculate y_actual
    predictions.dropna(inplace=True)                              # Drop all nas
    predictions['close'] = X_feature.iloc[split+1:,0].values      # Add close price from X_feature
    
    with col1:
        st.write(f"y_test data")
        st.dataframe(y_test)
    
    with col2:
        st.write(f"Predicted data")
        st.dataframe(predictions)

    days_bought=predictions[predictions['y_actual']== -1.0]
    st.write(f"Number of Days bought")
    st.dataframe(days_bought)

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
    #fig, ax = plt.subplots()
    entry_exit_plot = security_close *  entry * exit
    entry_exit_plot.opts(
        title="Short-Position Dual Moving Average Trading Algorithm"
    )
    #st.bokeh_chart(hv.render(entry_exit_plot, backend='bokeh'))
    st.write(hv.render(entry_exit_plot, backend='bokeh'))
    #st.write(fig, use_container_width=True)
    


    return




def main():

    #tickers = ['SPY', 'XLK', 'XLV', 'XLI', 'XLP', 'XLE', 'XLY', 'XLB']
    st.header("Sector Investing Strategy")
    select_value = ["Technology", "Health Care", "Industrial", "Consumer Staples", "Energy", "Consumer Discretionary", "iShares Core Canadian EFT"]
    sector = st.selectbox("Select the sector", select_value)

    if st.button("Submit"):
        st.text(f"Selection is {sector}")
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

        run_correlation(tick)
        run_model(tick)


if __name__ == '__main__':
    main()




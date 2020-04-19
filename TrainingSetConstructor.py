import pandas as pd
import numpy as np
import yfinance as yf
import SNode as sn
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_previous(data, date, lag):
    new_date = date - timedelta(days=lag)
    try:
        price = data[new_date.strftime("%Y-%m-%d")]
        if np.isnan(price):
            return price(data, date - timedelta(days=lag), 1)
        else:
            return price
    except:
        return np.nan


def get_change(data, date, lag, threshold):
    try:
        todays_price = data[date.strftime("%Y-%m-%d")]
    except:
        todays_price = get_previous(data, date, 1)

    if np.isnan(todays_price):
        todays_price = get_previous(data, date, 1)

    previous_price = get_previous(data, date, lag)
    min_change = todays_price*threshold
    if np.isnan(previous_price):
        return 0

    if previous_price >= todays_price + min_change:
        #technically 2 should be -1 but for the lagorithm later, it must be an actual index to work
        return 2
    if previous_price <= todays_price - min_change:
        return 1
    else:
        return 0

def get_dataset():
    MCD = yf.Ticker('MCD')
    YUM = yf.Ticker('YUM')
    QSR = yf.Ticker('QSR')
    DPZ = yf.Ticker('DPZ')

    since_date = "2020-04-08"


    target_words = {
        "MCD":{"search_terms":["mcdonalds", "macdonald", "macdonalds", "mcdonald", "big mac", "mc nuggets", "mickey D", "mickey d's", "mickey ds"],
               "stock_data": MCD.history(start=since_date)},
        "YUM":{"search_terms":["KFC", "kentucky fried chicken", "taco bell", "pizza hut"],
               "stock_data": YUM.history(start=since_date)},
        "QSR":{"search_terms":["tim hortons", "timmies", "tims", "popeyes", "burger king"],
               "stock_data": QSR.history(start=since_date)},
        "DPZ":{"search_terms":["domino's", "dominos"],
               "stock_data": DPZ.history(start=since_date)},
        }


    s_node = sn.SNode()
    s_node.load("node1.p")

    columns = []
    for key in target_words:
        columns.append(str(key))

    for key in target_words:
        columns.append( str(key) + "_open")
        columns.append(str(key) + "_y")

    df = pd.DataFrame(columns=columns)
    df["date"] = np.nan
    df = df.set_index('date')
    start_date = datetime.today() - timedelta(days=7)
    while start_date.day is not datetime.today().day:
        start = start_date
        end = start + timedelta(days=1)
        start = start.strftime("%Y-%m-%d")
        end = end.strftime("%Y-%m-%d")

        df.loc[start] = np.nan
        print("construting for: " + start )
        for key, value in target_words.items():
            sentiment = s_node.predict(value["search_terms"], 100, start, end)
            value["sentiment"] = sentiment
            df[key][start] = sentiment
            df[str(key)+"_y"][start] = get_change(value["stock_data"]["Open"], start_date, 3, 0.01)
            try:
                df[str(key)+"_open"][start] = value["stock_data"]["Open"][start]
            except:
                continue

        start_date += timedelta(days=1)

    for key in target_words:
        df[str(key) + "_y"] = df[str(key) + "_y"].shift(-3)

    print(df)
    df[target_words.keys()].plot()
    df[[column for column in list(df.columns) if "_y" in column]].plot()
    plt.show()
    return df

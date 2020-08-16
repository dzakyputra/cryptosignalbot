from telegram.ext import Updater, MessageHandler, Filters, CallbackContext
from poloniex import Poloniex

import plotly.graph_objects as go
import pandas as pd
import datetime
import time
import os

polo = Poloniex()


def get_chart_data(pair: str='USDT_BTC',
                   period: int=1800,
                   start: int=0,
                   end: int=int(time.time())):

    """Return the candlestick data with given parameters"""

    # Get period
    if period not in [300, 900, 1800, 7200, 14400, 86400]:
        return 'Please define the right period [300, 900, 1800, 7200, 14400, 86400]'
    
    # Get start and end time
    if start == 0:
        start = 7
    
    start_time = int((datetime.datetime.now() - datetime.timedelta(start)).timestamp())
    end_time = end

    # Get the ticker data from Poloniex
    ticker = polo.returnChartData(currencyPair=pair, 
                                  period=period, 
                                  start=start_time, 
                                  end=end_time)

    return ticker


def get_24h_volume():
    """Get the volume in the last 24 hours for all markets in USDT, return as a pandas dataframe"""

    # Get the pairs and filter it
    pairs = polo.return24hVolume()
    filtered_pair = list([key,value['USDT']] for key, value in pairs.items() if 'USDT_' in key)

    # Create a dataframe
    result = pd.DataFrame(filtered_pair, columns=['pair', 'volume']).sort_values(by='volume', ascending=False)
    result['volume'] = result['volume'].apply(human_format)

    return result


def save_chart_figure(df: pd.DataFrame(),
                      title: str=''):

    """Convert the ticker data (dataframe) into an image and save it as png"""

    avg_30 = df['close'].rolling(window=30, min_periods=1).mean()
    avg_50 = df['close'].rolling(window=50, min_periods=1).mean()

    # Visualizing using plotly
    fig = go.Figure()

    trace_candlestick = go.Candlestick(x=df['date'],
                                       open=df['open'],
                                       high=df['high'],
                                       low=df['low'],
                                       close=df['close'])

    trace_ma_30 = go.Scatter(x=df['date'], 
                             y=avg_30,
                             mode='lines',
                             line=dict(color='royalblue'))

    trace_ma_50 = go.Scatter(x=df['date'], 
                             y=avg_50,
                             mode='lines',
                             line=dict(color='firebrick'))

    fig.add_trace(trace_candlestick)
    fig.add_trace(trace_ma_30)
    fig.add_trace(trace_ma_50)

    fig.update_layout(title_text=title, title_x=0.5, xaxis_rangeslider_visible=False, showlegend=False)
    
    image_path = 'images/' + str(int(datetime.datetime.utcnow().timestamp())) + '-' + title + '.png'
    fig.write_image(image_path)

    return image_path

def signal_moving_average(df: pd.DataFrame(),
                          period: int):
    
    """Get the signal if MA 50 cross with the MA 30"""

    avg_30 = df['close'].rolling(window=30, min_periods=1).mean()
    avg_50 = df['close'].rolling(window=50, min_periods=1).mean()

    # 1. Check whether the MA 50 is under the MA 30 (it indicates uptrend)
    is_up = avg_50.tail(1).values[0] < avg_30.tail(1).values[0]
    
    # 2. Check how long is the current trend and find out whether the trend is about to change or not
    trend_time = 0
    crossed_price = 0
    differences = []

    for ma30, ma50, price in zip(avg_30[::-1], avg_50[::-1], df['close'][::-1]):
        if not ((ma50 < ma30) == is_up):
            crossed_price = price
            break

        # To find the length of trend
        trend_time += period

        # To find out whether the trend is about to change or not
        differences.append(abs(ma30 - ma50))
    
    current_price = df.iloc[-1]['close']
    trend_time = time.strftime('%H:%M:%S', time.gmtime(trend_time))

    return {
        'is_uptrend': is_up,
        'trend_time': trend_time,
        'current_price': current_price,
        'crossed_price': crossed_price,
        'differences': differences
    }

def signal_break_support_resist(df: pd.DataFrame()):
    
    """Get the signal whether the chart is break resist (uptrend) or break support (downtrend)"""

    # Since we don't want to a large window, we take the last 5 hours of windows transaction only
    prices = df['close'].tail(60).tolist()

    # Get the necessary parameters
    start_price = prices[0]
    current_price = prices.pop()
    min_price = min(prices)
    max_price = max(prices)

    return {
        'is_break_resist': current_price > max_price,
        'is_break_support': current_price < min_price,
        'start_price': start_price,
        'current_price': current_price,
        'min_price': min_price,
        'max_price': max_price
    }

def signal_suddenly_increase(df: pd.DataFrame(), threshold):

    """Get the percentages differences between the last three candles"""

    # Get the last three candles, then calculate the differences
    tails = df['close'].tail(3).tolist()
    differences = ((tails[-1] - tails[0]) / tails[0]) * 100 
    is_suddenly_increase = differences > threshold

    return differences, is_suddenly_increase

def convert_chart_data_into_dataframe(chart_data):
    
    """Convert the ticker from polinex API into pandas dataframe"""

    # Convert to a dataframe and change the date data type
    df = pd.DataFrame(chart_data)
    df['date'] = pd.to_datetime(df['date'], unit='s')

    return df

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '$ {}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', ' K', ' M', ' B', ' T'][magnitude])


# chart_data = get_chart_data('USDT_ETH', 300, 1)
# df = convert_chart_data_into_dataframe(chart_data)

# print(signal_suddenly_increase(df, 1))

pairs = get_24h_volume()
top_20_pairs = pairs.head(10)['pair'].tolist()

for pair in top_20_pairs:
    
    chart_data = get_chart_data(pair, 1800, 1)
    df = convert_chart_data_into_dataframe(chart_data)

    print(pair, signal_suddenly_increase(df, 1))

#print([k for k in pairs.keys() if 'USDT_' in k])

# def callback_minute(context: CallbackContext):
#     print('heii')

# def send_the_image(update, context):

#     """Return a chart image based on the given pair and time period"""

#     # Get the pair, period, and day
#     keyword = update.message.text.split()
#     pair, period, day = 'USDT_{}'.format(keyword[0].upper()), int(keyword[1])*60, int(keyword[2])

#     # Convert it into dataframe
#     chart_data = get_chart_data(pair, period, day)
#     df = convert_chart_data_into_dataframe(chart_data)

#     # Create the chart and send it
#     image_path = save_chart_figure(df, pair)
#     context.bot.send_photo(chat_id=update.message.chat_id, photo=open(image_path, 'rb'))
#     os.remove(image_path)

# def main():
#     updater = Updater('1300547914:AAHvtAa5jof_5QZwyYtnbsHRcWaAooRdsj8', use_context=True)
#     job = updater.job_queue

#     dp = updater.dispatcher
#     dp.add_handler(MessageHandler(Filters.text, send_the_image))
    
#     job_minute = job.run_repeating(callback_minute, interval=10, first=0)

#     updater.start_polling()
#     updater.idle()

# if __name__ == '__main__':
#     main()
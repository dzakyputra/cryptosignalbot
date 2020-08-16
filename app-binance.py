from telegram.ext import Updater, MessageHandler, Filters, CallbackContext
from binance.client import Client

import plotly.graph_objects as go
import pandas as pd
import datetime
import telegram
import time
import os


client = Client('BINANCE_PUBLIC_KEY', 'BINANCE_PRIVATE_KEY')


def get_chart_data(pair: str = 'BTCUSDT', 
                   interval: str = Client.KLINE_INTERVAL_15MINUTE,
                   period: str = '1 day'):

    # Get the candlestick data and convert it into a dataframe
    klines = client.get_historical_klines(pair, interval, '{} ago UTC'.format(period))

    return klines


def save_chart_figure(df: pd.DataFrame(),
                      title: str=''):

    """Convert the ticker data (dataframe) into an image and save it as png"""

    avg_30 = df['close'].rolling(window=30, min_periods=1).mean()
    avg_50 = df['close'].rolling(window=50, min_periods=1).mean()

    # Visualizing using plotly
    fig = go.Figure()

    trace_candlestick = go.Candlestick(x=df['close_time'],
                                       open=df['open'],
                                       high=df['high'],
                                       low=df['low'],
                                       close=df['close'])

    trace_ma_30 = go.Scatter(x=df['close_time'], 
                             y=avg_30,
                             mode='lines',
                             line=dict(color='royalblue'))

    trace_ma_50 = go.Scatter(x=df['close_time'], 
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


def convert_chart_data_into_dataframe(klines):
    
    """Convert the ticker from Binance API into pandas dataframe"""

    # Convert the data into dataframe
    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'volume', 'trades', 'taker_base', 'taker_quote', 'ignore'])
    
    # Convert the timestamp into human readable format
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    
    return df


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

    if len(tails) < 3:
        return 0, False

    differences = ((tails[-1] - tails[0]) / tails[0]) * 100 
    is_suddenly_increase = differences > threshold

    return {
        'current_price': tails[-1],
        'base_price': tails[0],
        'differences': differences,
        'is_suddenly_increase': is_suddenly_increase
    }

def screening_1_minute(context: CallbackContext):
    exception = ['USDTBIDR', 'USDTIDRT', 'USDTBKRW']

    tickers = client.get_ticker()
    USDT_market = [pair['symbol'] for pair in tickers if ('USDT' in pair['symbol'] and pair['symbol'] not in exception)]
    markets = pd.DataFrame(tickers)
    markets = markets[markets['symbol'].isin(USDT_market)]
    markets['quoteVolume'] = markets['quoteVolume'].astype(float)
    markets = markets.sort_values('quoteVolume', ascending=False).head(30)['symbol'].tolist()

    all_suddenly_increase = []
    all_break_support_resist = []
    all_moving_average = []

    message_suddenly_increase = '*SIGNAL - Suddenly Increase* \n\n'
    message_break_support_resist = '*SIGNAL - Break Resist* \n\n'
    message_moving_average = '*SIGNAL - Moving Average* \n\n'

    for pair in markets:
        chart_data = get_chart_data(pair, '5m', '1 day')
        df = convert_chart_data_into_dataframe(chart_data)

        result_suddenly_increase = signal_suddenly_increase(df, 2)
        if result_suddenly_increase['is_suddenly_increase']:
            message_suddenly_increase += pair + ' | ' + '{:.2f}%'.format(result_suddenly_increase['differences']) + '\n'
            all_suddenly_increase.append([pair, result_suddenly_increase['differences'], result_suddenly_increase['base_price'], result_suddenly_increase['current_price']])

        result_break_support_resist = signal_break_support_resist(df)
        if result_break_support_resist['is_break_resist']:
            message_break_support_resist += pair + '\n'
            all_break_support_resist.append([pair, result_break_support_resist['current_price'], result_break_support_resist['max_price']])
        
        result_moving_average = signal_moving_average(df, 300)
        if (result_moving_average['is_uptrend'] and len(result_moving_average['differences']) <= 3):
            message_moving_average += pair + '\n'
            all_moving_average.append(pair)

    if len(all_suddenly_increase) > 0:
        context.bot.send_message(chat_id='TELEGRAM_USER_ID', 
                                 text=message_suddenly_increase,
                                 parse_mode=telegram.ParseMode.MARKDOWN)

    if len(all_break_support_resist) > 0:
        context.bot.send_message(chat_id='TELEGRAM_USER_ID', 
                                 text=message_break_support_resist,
                                 parse_mode=telegram.ParseMode.MARKDOWN)
    
    if len(all_moving_average) > 0:
        context.bot.send_message(chat_id='TELEGRAM_USER_ID', 
                                 text=message_moving_average,
                                 parse_mode=telegram.ParseMode.MARKDOWN)

def send_the_image(update, context):

    """Return a chart image based on the given pair and time period"""

    # Get the pair, period, and day
    keyword = update.message.text.split()
    if len(keyword) == 1:
        pair, interval, period = keyword[0].upper(), '5m', '1 day'
    else:
        pair, interval, period = keyword[0].upper(), keyword[1], keyword[2] + ' ' + keyword[3]

    # Convert it into dataframe
    chart_data = get_chart_data(pair, interval, period)
    df = convert_chart_data_into_dataframe(chart_data)

    # Create the chart and send it
    image_path = save_chart_figure(df, pair)
    context.bot.send_photo(chat_id=update.message.chat_id, photo=open(image_path, 'rb'))
    os.remove(image_path)

def main():
    updater = Updater('TELEGRAM_BOT_TOKEN', use_context=True)
    job = updater.job_queue

    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.text, send_the_image))
    
    job.run_repeating(screening_1_minute, interval=300, first=0)

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
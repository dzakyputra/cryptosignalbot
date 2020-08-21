from telegram.ext import Updater, MessageHandler, Filters, CallbackContext
from binance.client import Client

import plotly.graph_objects as go
import pandas as pd
import datetime
import telegram
import sqlite3
import time
import os


BINANCE_PUBLIC_KEY = os.getenv('BINANCE_PUBLIC_KEY')
BINANCE_PRIVATE_KEY = os.getenv('BINANCE_PRIVATE_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_USER_ID = os.getenv('TELEGRAM_USER_ID')
SQLITE_DATABASE_PATH = os.getenv('SQLITE_DATABASE_PATH')


class Signal:

    def __init__(self, pair: str, chart_data: pd.DataFrame):
        self.pair = pair
        self.chart_data = chart_data
        self.timestamp = chart_data.tail(1)['close_time'].dt.strftime('%Y-%m-%d %H:%m:%S').values[0]

    def crossovers_moving_average(self, short_period: int, long_period: int, period: int):

        """Get the signal if the longer MA period cross with the shorter MA period"""

        short_period_ma = self.chart_data['close'].rolling(window=short_period, min_periods=1).mean()
        long_period_ma = self.chart_data['close'].rolling(window=long_period, min_periods=1).mean()

        # 1. Check whether the longer MA period is under the shorter MA period (it indicates uptrend)
        is_up = long_period_ma.tail(1).values[0] < short_period_ma.tail(1).values[0]
        
        # 2. Check how long is the current trend and find out whether the trend is about to change or not
        trend_time = 0
        crossed_price = 0
        differences = []

        for short_period, long_period, price in zip(short_period_ma[::-1], long_period_ma[::-1], self.chart_data['close'][::-1]):
            if not ((long_period < short_period) == is_up):
                crossed_price = price
                break

            # To find the length of trend
            trend_time += period

            # To find out whether the trend is about to change or not
            differences.append(abs(short_period - long_period))
        
        current_price = self.chart_data.iloc[-1]['close']
        trend_time = time.strftime('%H:%M:%S', time.gmtime(trend_time))

        # Determine if the signal will be sent or not
        is_send = is_up and (len(differences) <= 3)

        return {
            'title': 'Crossovers Moving Average',
            'is_send': is_send,
            'is_uptrend': is_up,
            'trend_time': trend_time,
            'current_price': current_price,
            'crossed_price': crossed_price,
            'differences': differences,
            'timestamp': self.timestamp
        }

    def break_support_resistance(self, window: int):

        """Get the signal whether the chart is break resistance (uptrend) or break support (downtrend)"""

        # Get the prices based on the given window
        prices = self.chart_data['close'].tail(window).tolist()

        # Get the necessary parameters
        start_price = prices[0]
        current_price = prices.pop()
        min_price = min(prices)
        max_price = max(prices)

        # Determine if the signal will be sent or not
        is_send = current_price > max_price

        return {
            'title': 'Break Support Resistance',
            'is_send': is_send,
            'is_break_resist': current_price > max_price,
            'is_break_support': current_price < min_price,
            'start_price': start_price,
            'current_price': current_price,
            'min_price': min_price,
            'max_price': max_price,
            'timestamp': self.timestamp
        }

    def price_change(self, window: int, threshold: int):

        """Give a signal if a price rises above / below certain threshold (percentage) in the given period of time"""

         # Get the last (window) candlestick data, then calculate the differences
        tails = self.chart_data['close'].tail(window).tolist()

        if len(tails) < window:
            return 0, False

        differences = ((tails[-1] - tails[0]) / tails[0]) * 100 
        is_above_threshold = differences > threshold
        is_below_threshold = differences < -threshold

        # Determine if the signal will be sent or not
        is_send = is_above_threshold

        return {
            'title': 'Price Change',
            'is_send': is_send,
            'is_above_threshold': is_above_threshold,
            'is_below_threshold': is_below_threshold,
            'current_price': tails[-1],
            'base_price': tails[0],
            'differences': differences,
            'timestamp': self.timestamp
        }


class Binance:

    def __init__(self, binance_public_key, binance_private_key):
        self.client = Client(binance_public_key, binance_private_key)

    def get_chart_data(self, pair: str = 'BTCUSDT', interval: str = Client.KLINE_INTERVAL_15MINUTE, period: str = '1 day'):

        # Get the candlestick data and convert it into a dataframe
        klines = self.client.get_historical_klines(pair, interval, '{} ago UTC'.format(period))

        return klines

    def convert_chart_data_into_dataframe(self, klines):
    
        """Convert the ticker from Binance API into pandas dataframe"""

        # Convert the data into dataframe
        df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'volume', 'trades', 'taker_base', 'taker_quote', 'ignore'])
        
        # Convert the timestamp into human readable format
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df['close'] = df['close'].astype(float)
        
        return df

    def save_chart_figure(self, df: pd.DataFrame, title: str=''):

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
        
        # Save the visualization
        image_path = 'images/' + str(int(datetime.datetime.utcnow().timestamp())) + '-' + title + '.png'
        fig.write_image(image_path)

        return image_path


class Database:

    def __init__(self, db_file: str):
        self.conn = None
        try:
            self.conn = sqlite3.connect(db_file)
        except Error as e:
            print(e)

    def insert_signal_data(self, data: dict):

        """Insert the signal data into the sqlite database"""

        # Define the signal's table characteristic
        signal_references = {
            'PRICE CHANGE': {
                'table': 'signal_pc',
                'total_columns': 7
            },

            'BREAK SUPPORT RESISTANCE': {
                'table': 'signal_bsr',
                'total_columns': 8
            },

            'CROSSOVERS MOVING AVERAGE': {
                'table': 'signal_cma',
                'total_columns': 6
            }
        }

        # For each signal, we want to store the given data in the database
        for signal_name, signal_data in data.items():

            if len(signal_data) > 0:
                
                total_values = '(' + ',?'*signal_references[signal_name]['total_columns'] + ')'
                total_values = '(' + total_values[2:]
                table_name = signal_references[signal_name]['table']

                sql = ''' 
                        INSERT INTO 
                            {}
                        VALUES
                            {}
                    '''.format(table_name, total_values)

                cur = self.conn.cursor()
                cur.executemany(sql, signal_data)
                print(cur.rowcount, 'records have been inserted to the table.')

                self.conn.commit()

        self.conn.close()


# Init the Binance Class
binance = Binance(binance_public_key=BINANCE_PUBLIC_KEY, binance_private_key=BINANCE_PRIVATE_KEY)


class Message:

    def __init__(self, pairs: list, signals: dict):
        self.pairs = pairs
        self.signals = signals

    def compose_message(self):

        """Gather the signals from given pairs and send them to the given telegram user id"""

        # Prepare the message to be send and the data for the database
        message = {}
        data = {}
        for name, parameters in self.signals.items():
            message[' '.join(name.split('_')).upper()] = []
            data[' '.join(name.split('_')).upper()] = []

        for pair in self.pairs:

            # Get the chart data for the given pair and the last day on 5 minutes period
            chart_data = binance.get_chart_data(pair, '5m', '1 day')
            chart_data = binance.convert_chart_data_into_dataframe(chart_data)

            # Initiate the Signal class and run the signal method based on the signal name
            signal = Signal(pair, chart_data)

            for name, parameters in self.signals.items():

                if name == 'break_support_resistance':
                    signal_result = signal.break_support_resistance(parameters['window'])
                    if signal_result['is_send']:
                        data['BREAK SUPPORT RESISTANCE'].append((signal_result['timestamp'],
                                                                pair,
                                                                signal_result['is_break_resist'],
                                                                signal_result['is_break_support'],
                                                                signal_result['start_price'],
                                                                signal_result['current_price'],
                                                                signal_result['min_price'],
                                                                signal_result['max_price']))

                elif name == 'crossovers_moving_average':
                    signal_result = signal.crossovers_moving_average(parameters['short_period'], parameters['long_period'], parameters['period'])
                    if signal_result['is_send']:
                        data['CROSSOVERS MOVING AVERAGE'].append((signal_result['timestamp'],
                                                                  pair,
                                                                  signal_result['is_uptrend'],
                                                                  signal_result['trend_time'],
                                                                  signal_result['current_price'],
                                                                  len(signal_result['differences'])))

                elif name == 'price_change':
                    signal_result = signal.price_change(parameters['window'], parameters['threshold'])
                    if signal_result['is_send']:
                        data['PRICE CHANGE'].append((signal_result['timestamp'],
                                                     pair,
                                                     signal_result['is_above_threshold'],
                                                     signal_result['is_below_threshold'],
                                                     signal_result['current_price'],
                                                     signal_result['base_price'],
                                                     signal_result['differences']))

                if signal_result['is_send']:
                    message[' '.join(name.split('_')).upper()].append(pair)

        # Save the signal data into the database
        database = Database(db_file=SQLITE_DATABASE_PATH)
        database.insert_signal_data(data=data)

        # Compose the message to be send
        send = ''
        for signal_name, pairs in message.items():
            if len(pairs) > 0:
                send += '\n*[' + ' '.join(signal_name.split()).upper() + ']*\n\n'
                for pair in pairs:
                    send += pair + '\n'

        return send


def screening_1_minute(context: CallbackContext):

    """Every one minute, this function will run and send the notification to the given user"""

    # Set the exception for pairs we do not want to see
    exception = ['USDTBIDR', 'USDTIDRT', 'USDTBKRW', 'USDCUSDT']

    # Get the list of the USDT pair
    tickers = binance.client.get_ticker()
    USDT_market = [pair['symbol'] for pair in tickers if ('USDT' in pair['symbol'] and pair['symbol'] not in exception)]
    markets = pd.DataFrame(tickers)
    markets = markets[markets['symbol'].isin(USDT_market)]
    
    # Sort the pairs by volume and get the top 30 highest volume
    markets['quoteVolume'] = markets['quoteVolume'].astype(float)
    markets = markets.sort_values('quoteVolume', ascending=False).head(30)['symbol'].tolist()

    # Signals we are going to use
    signals = {
        'price_change': {
            'window': 3,
            'threshold': 2
        },

        'break_support_resistance': {
            'window': 60
        },

        'crossovers_moving_average': {
            'short_period': 30,
            'long_period': 50,
            'period': 300
        }
    }

    # Compose the message and send it
    message = Message(pairs=markets, signals=signals)
    message = message.compose_message()

    context.bot.send_message(chat_id=TELEGRAM_USER_ID,
                             text=message,
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
    chart_data = binance.get_chart_data(pair, interval, period)
    df = binance.convert_chart_data_into_dataframe(chart_data)

    # Create the chart and send it
    image_path = binance.save_chart_figure(df, pair)
    context.bot.send_photo(chat_id=update.message.chat_id, photo=open(image_path, 'rb'))
    os.remove(image_path)


def main():
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    job = updater.job_queue

    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.text, send_the_image))
    
    job.run_repeating(screening_1_minute, interval=300, first=0)

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
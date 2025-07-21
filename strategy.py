import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import time
import requests
import telebot
import os

TELEGRAM_TOKEN = '7393748470:AAEFTKeMexVrbstjw3vyHI71-lHOPC0WSIc'
TELEGRAM_CHAT_ID = '821584974'
ALERT_COOLDOWN = 30  # Минут между сигналами для одного символа

TOKEN = os.getenv("TOKEN")  # заберёт из переменных окружения
bot = telebot.TeleBot(TOKEN)

symbols = ['BTC/USDT', 'XRP/USDT', 'LTC/USDT', 'ADA/USDT','LINK/USDT','DOGE/USDT', 'OP/USDT', '1000PEPE/USDT', 'SUI/USDT']
timeframe = '1m'
limit = 500000
SWING_WINDOW = 3
VOLUME_MULTIPLIER = 1.8
MAX_HOLD_MINUTES = 30
INITIAL_BALANCE = 18
RISK_PCT = 10
MIN_ADX = 25
CONFIRMATION_BARS = 2

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    params = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        response = requests.post(url, params=params)
        return response.json()
    except Exception as e:
        print(f"Ошибка отправки в Telegram: {e}")

def is_swing_high(highs, i, window=SWING_WINDOW):
    start = max(0, i - window)
    end = min(len(highs), i + window + 1)
    return highs[i] == max(highs[start:end])

def is_swing_low(lows, i, window=SWING_WINDOW):
    start = max(0, i - window)
    end = min(len(lows), i + window + 1)
    return lows[i] == min(lows[start:end])

def detect_swings(df):
    df['swing_high'] = np.nan
    df['swing_low'] = np.nan
    highs = df['high'].values
    lows = df['low'].values
    
    for i in range(len(df)):
        if is_swing_high(highs, i):
            df.loc[df.index[i], 'swing_high'] = highs[i]
        if is_swing_low(lows, i):
            df.loc[df.index[i], 'swing_low'] = lows[i]
            
    # Исправление предупреждения pandas
    df = df.copy()
    df['swing_high'] = df['swing_high'].ffill()
    df['swing_low'] = df['swing_low'].ffill()
    return df

def calculate_indicators(df):
    # ATR (Average True Range)
    df['tr0'] = df['high'] - df['low']
    df['tr1'] = abs(df['high'] - df['close'].shift())
    df['tr2'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()
    
    # EMA (Exponential Moving Average)
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Volume MA
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_spike'] = df['volume'] > (df['vol_ma'] * VOLUME_MULTIPLIER)
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Удаляем временные столбцы
    df.drop(['tr0', 'tr1', 'tr2', 'tr'], axis=1, inplace=True, errors='ignore')
    
    return df

def find_fvg(df):
    df['fvg_up'] = False
    df['fvg_down'] = False
    
    for i in range(2, len(df)):
        # Восходящий FVG (разрыв вверх)
        if df['low'].iloc[i] > max(df['high'].iloc[i-2], df['high'].iloc[i-1]):
            df.loc[df.index[i], 'fvg_up'] = True
        
        # Нисходящий FVG (разрыв вниз)
        if df['high'].iloc[i] < min(df['low'].iloc[i-2], df['low'].iloc[i-1]):
            df.loc[df.index[i], 'fvg_down'] = True
            
    return df

def is_bullish_engulfing(df, i):
    if i < 1 or i >= len(df):
        return False
    prev = df.iloc[i-1]
    curr = df.iloc[i]
    
    # Упрощенные условия для тестирования
    return (curr['close'] > curr['open'] and 
            prev['close'] < prev['open'] and 
            curr['open'] < prev['close'] and 
            curr['close'] > prev['open'])

def is_bearish_engulfing(df, i):
    if i < 1 or i >= len(df):
        return False
    prev = df.iloc[i-1]
    curr = df.iloc[i]
    
    # Упрощенные условия для тестирования
    return (curr['close'] < curr['open'] and 
            prev['close'] > prev['open'] and 
            curr['open'] > prev['close'] and 
            curr['close'] < prev['open'])

def detect_market_structure(df):
    df = df.copy()
    df['BOS_up'] = False
    df['BOS_down'] = False
    df['CHoCH_up'] = False
    df['CHoCH_down'] = False
    
    trend = None
    last_high = df['swing_high'].iloc[0] if not np.isnan(df['swing_high'].iloc[0]) else df['high'].iloc[0]
    last_low = df['swing_low'].iloc[0] if not np.isnan(df['swing_low'].iloc[0]) else df['low'].iloc[0]
    
    for i in range(1, len(df)):
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        current_close = df['close'].iloc[i]
        
        # Обновляем последние экстремумы при появлении новых свингов
        if not np.isnan(df['swing_high'].iloc[i]):
            last_high = df['swing_high'].iloc[i]
        if not np.isnan(df['swing_low'].iloc[i]):
            last_low = df['swing_low'].iloc[i]
        
        # Определяем прорывы (BOS)
        if current_close > last_high:
            if trend != 'up':
                df.loc[df.index[i], 'BOS_up'] = True
                trend = 'up'
        elif current_close < last_low:
            if trend != 'down':
                df.loc[df.index[i], 'BOS_down'] = True
                trend = 'down'
                
        # Определяем смену тренда (CHoCH)
        if trend == 'down' and current_close > last_high:
            df.loc[df.index[i], 'CHoCH_up'] = True
            trend = 'up'
        elif trend == 'up' and current_close < last_low:
            df.loc[df.index[i], 'CHoCH_down'] = True
            trend = 'down'
            
    return df

def simulate_trading(df, symbol, initial_balance=INITIAL_BALANCE, risk_pct=RISK_PCT):
    in_trade = False
    trades = []
    balance = initial_balance
    last_trade_time = None
    COOLDOWN_MINUTES = 5

    stats = {
        'signals': 0,
        'entries': 0,
        'wins': 0,
        'losses': 0,
        'missed_cooldown': 0,
        'missed_time_filter': 0,
    }

    for i in range(30, len(df)):
        row = df.iloc[i]
        dt = df.index[i]
        prev_row = df.iloc[i - 1]

        # Пропуск крайних минут часа
        if 0 <= dt.minute < 5 or 55 <= dt.minute <= 59:
            stats['missed_time_filter'] += 1
            continue

        # Выход из сделки
        if in_trade:
            duration = (dt - entry_time).total_seconds() / 60
            exit_price = row['close']
            sl_hit = (row['low'] <= sl) if position_type == 'long' else (row['high'] >= sl)
            tp_hit = (row['high'] >= tp) if position_type == 'long' else (row['low'] <= tp)

            # Трейлинг SL
            if position_type == 'long':
                new_sl = max(sl, row['high'] - 1.5 * row['atr'])
                if new_sl > sl:
                    sl = new_sl
            else:
                new_sl = min(sl, row['low'] + 1.5 * row['atr'])
                if new_sl < sl:
                    sl = new_sl

            if sl_hit or tp_hit or duration > MAX_HOLD_MINUTES:
                if tp_hit:
                    result = 'TP'
                    exit_price = tp
                    stats['wins'] += 1
                elif sl_hit:
                    result = 'SL'
                    exit_price = sl
                    stats['losses'] += 1
                else:
                    result = 'TIME'
                    stats['losses'] += 1

                price_diff = exit_price - entry if position_type == 'long' else entry - exit_price
                pnl_pct = (price_diff / entry) * 100
                dollar_risk = balance * (risk_pct / 100)
                risk_per_share = abs(entry - initial_sl)
                position_size = dollar_risk / risk_per_share
                pnl_dollar = position_size * price_diff
                balance += pnl_dollar

                trades[-1].update({
                    'result': result,
                    'exit': exit_price,
                    'duration': round(duration, 2),
                    'pnl_pct': round(pnl_pct, 2),
                    'pnl_dollar': round(pnl_dollar, 2),
                    'balance_after': round(balance, 2)
                })
                in_trade = False
                continue

        # Упрощенные сигналы входа:
        trend_up = row['close'] > row['ema_50'] and row['ema_20'] > row['ema_50']
        trend_down = row['close'] < row['ema_50'] and row['ema_20'] < row['ema_50']

        buy_signal = (prev_row['BOS_up'] or prev_row['CHoCH_up']) and trend_up
        sell_signal = (prev_row['BOS_down'] or prev_row['CHoCH_down']) and trend_down

        if buy_signal or sell_signal:
            stats['signals'] += 1
            print(f"{dt} | {symbol} | Сигнал: {'BUY' if buy_signal else 'SELL'}")

        if last_trade_time is not None:
            minutes_since_last = (dt - last_trade_time).total_seconds() / 60
            if minutes_since_last < COOLDOWN_MINUTES:
                buy_signal = False
                sell_signal = False
                stats['missed_cooldown'] += 1

        if buy_signal and not in_trade:
            entry = row['close']
            risk = max(0.5, min(1.5, 1.0 - (row['atr'] / entry))) * row['atr']
            sl = entry - risk
            tp = entry + 2.5 * risk
            initial_sl = sl
            entry_time = dt
            position_type = 'long'
            in_trade = True
            last_trade_time = dt
            stats['entries'] += 1

            trades.append({
                'symbol': symbol,
                'time': dt,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp': round(tp, 4),
                'type': 'long',
                'result': 'OPEN',
                'atr': round(row['atr'], 4)
            })

        if sell_signal and not in_trade:
            entry = row['close']
            risk = max(0.5, min(1.5, 1.0 - (row['atr'] / entry))) * row['atr']
            sl = entry + risk
            tp = entry - 2.5 * risk
            initial_sl = sl
            entry_time = dt
            position_type = 'short'
            in_trade = True
            last_trade_time = dt
            stats['entries'] += 1

            trades.append({
                'symbol': symbol,
                'time': dt,
                'entry': round(entry, 4),
                'sl': round(sl, 4),
                'tp': round(tp, 4),
                'type': 'short',
                'result': 'OPEN',
                'atr': round(row['atr'], 4)
            })

    return trades, balance, stats


def monitor_market():
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })
    
    # Инициализируем с осведомленными датами UTC
    last_alert_times = {symbol: datetime(1970, 1, 1, tzinfo=timezone.utc) for symbol in symbols}
    
    while True:
        try:
            current_time = datetime.now(timezone.utc)
            print(f"\n🔍 Цикл проверки: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Пропускаем первые/последние 5 минут часа
            if 0 <= current_time.minute < 5 or 55 <= current_time.minute <= 59:
                print("⏸️ Пропуск неактивного периода")
                time.sleep(60)
                continue
                
            for symbol in symbols:
                try:
                    # Проверяем кулдаун
                    minutes_since_last = (current_time - last_alert_times[symbol]).total_seconds() / 60
                    if minutes_since_last < ALERT_COOLDOWN:
                        continue
                    
                    # Загрузка последних данных
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                    df.set_index('datetime', inplace=True)
                    
                    # Обработка данных
                    df = detect_swings(df)
                    df = detect_market_structure(df)
                    df = calculate_indicators(df)
                    df = find_fvg(df)
                    
                    # Анализ последней свечи
                    last_index = df.index[-1]
                    last_row = df.iloc[-1]
                    prev_row = df.iloc[-2]
                    
                    # Определение тренда
                    trend_up = last_row['close'] > last_row['ema_50'] and last_row['ema_20'] > last_row['ema_50']
                    trend_down = last_row['close'] < last_row['ema_50'] and last_row['ema_20'] < last_row['ema_50']
                    
                    # Сигналы
                    buy_signal = (prev_row['BOS_up'] or prev_row['CHoCH_up']) and trend_up
                    sell_signal = (prev_row['BOS_down'] or prev_row['CHoCH_down']) and trend_down
                    
                    # Формирование сообщения с TP/SL
                    if buy_signal or sell_signal:
                        entry_price = last_row['close']
                        atr = last_row['atr']
                        
                        if buy_signal:
                            sl = entry_price - 1.5 * atr  # Stop Loss ниже на 1.5 ATR
                            tp = entry_price + 3.0 * atr  # Take Profit выше на 3 ATR
                            position_type = "LONG"
                        else:
                            sl = entry_price + 1.5 * atr  # Stop Loss выше на 1.5 ATR
                            tp = entry_price - 3.0 * atr  # Take Profit ниже на 3 ATR
                            position_type = "SHORT"
                        
                        risk_reward = abs((tp - entry_price) / (entry_price - sl)) if buy_signal else abs((entry_price - tp) / (sl - entry_price))
                        
                        message = (
                            f"🚀 <b>Торговый сигнал: {symbol}</b>\n"
                            f"📊 Тип позиции: {position_type}\n"
                            f"💰 Цена входа: {entry_price:.6f}\n"
                            f"🛑 Stop Loss: {sl:.6f}\n"
                            f"🎯 Take Profit: {tp:.6f}\n"
                            f"📈 Risk/Reward: 1:{risk_reward:.1f}\n"
                            f"📶 RSI: {last_row.get('rsi', 'N/A'):.2f}\n"
                            f"🕒 Время: {last_index.strftime('%Y-%m-%d %H:%M')} UTC"
                        )
                        
                        send_telegram_message(message)
                        last_alert_times[symbol] = current_time
                        print(f"📨 Отправлен сигнал для {symbol}")
                        
                except Exception as e:
                    print(f"⚠️ Ошибка для {symbol}: {str(e)}")
            
            # Пауза до следующей минуты
            now_utc = datetime.now(timezone.utc)
            sleep_time = 60 - now_utc.second
            if sleep_time < 0:
                sleep_time = 5
            print(f"⏳ Следующая проверка через {sleep_time} сек.")
            time.sleep(sleep_time)
            
        except Exception as e:
            print(f"🔥 Критическая ошибка: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    print("Запуск мониторинга рынка в реальном времени...")
    



def print_performance_report(trades, stats, symbol, balance):
    """Генерация детального отчета о производительности"""
    print(f"\n📊 Отчет по {symbol}")
    print(f"  Конечный баланс: ${balance:.2f} ({balance - INITIAL_BALANCE:.2f} {'+' if balance > INITIAL_BALANCE else ''})")
    print(f"  Всего сделок: {len(trades)}")
    
    if trades:
        win_trades = [t for t in trades if t.get('result') == 'TP']
        loss_trades = [t for t in trades if t.get('result') in ['SL', 'TIME']]
        
        win_rate = len(win_trades) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t['pnl_pct'] for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in loss_trades]) if loss_trades else 0
        profit_factor = abs(sum(t['pnl_dollar'] for t in win_trades) / 
                          sum(abs(t['pnl_dollar']) for t in loss_trades)) if loss_trades else float('inf')
        
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Средняя прибыль: {avg_win:.2f}% | Средний убыток: {avg_loss:.2f}%")
        print(f"  Фактор прибыли: {profit_factor:.2f}")
        
        # Рассчитываем максимальную просадку
        min_balance = min(t['balance_after'] for t in trades)
        max_drawdown = min_balance - INITIAL_BALANCE
        print(f"  Максимальная просадка: {max_drawdown:.2f}")
        
        # Анализ по типам сделок
        long_trades = [t for t in trades if t['type'] == 'long']
        short_trades = [t for t in trades if t['type'] == 'short']
        
        if long_trades:
            long_win_rate = len([t for t in long_trades if t.get('result') == 'TP']) / len(long_trades) * 100
            print(f"  Лонги: {len(long_trades)} сделок, Win Rate: {long_win_rate:.1f}%")
        
        if short_trades:
            short_win_rate = len([t for t in short_trades if t.get('result') == 'TP']) / len(short_trades) * 100
            print(f"  Шорты: {len(short_trades)} сделок, Win Rate: {short_win_rate:.1f}%")
    
    print("\n📈 Статистика сигналов:")
    print(f"  Всего сигналов: {stats['signals']}")
    print(f"  Исполненных входов: {stats['entries']}")
    print(f"  Пропущено (кулдаун): {stats['missed_cooldown']}")
    print(f"  Пропущено (время): {stats['missed_time_filter']}")

# MAIN
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
    'rateLimit': 3000
})

# Загружаем данные для всех символов сразу
def load_data(symbols, timeframe, days_back=3):
    data = {}
    # Исправление deprecated utcnow
    since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days_back)).strftime('%Y-%m-%dT%H:%M:%S'))
    
    for symbol in symbols:
        print(f"Загрузка данных для {symbol}...")
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            data[symbol] = df
        except Exception as e:
            print(f"Ошибка загрузки {symbol}: {str(e)}")
    
    return data

print("Начало улучшенной стратегии скальпинга")
# Исправление deprecated utcnow
print(f"Дата запуска: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Символы: {', '.join(symbols)}")
print(f"Таймфрейм: {timeframe}")
print(f"Начальный баланс: ${INITIAL_BALANCE}")
print(f"Риск на сделку: {RISK_PCT}%\n")

# Загружаем данные
all_data = load_data(symbols, timeframe, days_back=3)

# Обрабатываем и торгуем
for symbol, df in all_data.items():
    if df.empty:
        continue
        
    print(f"\n🚀 Анализ: {symbol}")
    try:
        # Обработка данных
        print("   🔄 Обработка данных...")
        df = detect_swings(df)
        df = detect_market_structure(df)
        df = calculate_indicators(df)
        df = find_fvg(df)
        
        # Симуляция торговли
        print("   ⚙️ Симуляция торговли...")
        trades, final_balance, stats = simulate_trading(df, symbol)
        
        # Результаты
        print(f"   ✅ Результат: {len(trades)} сделок | Баланс: ${final_balance:.2f}")
        
        # Детальный отчет
        print_performance_report(trades, stats, symbol, final_balance)

    except Exception as e:
        print(f"   ❌ Ошибка: {str(e)}")

print("\nСтратегия завершена!")



@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Привет, я на Heroku!")

bot.infinity_polling()
monitor_market()
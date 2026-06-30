import math
import numpy as np
import ccxt
import streamlit as st
from scipy.optimize import minimize_scalar
import time
import json
import os

# Настройка адаптивного интерфейса под ПК и смартфоны
st.set_page_config(
    page_title="Arbitrage Crypto Bot",
    page_icon="⚡",
    layout="centered"
)

# ---------- Файлы для хранения ----------
KEYS_FILE = "api_keys.json"
FEES_FILE = "fees.json"

# ---------- Загрузка/сохранение настроек ----------
def load_api_keys():
    if os.path.exists(KEYS_FILE):
        try:
            with open(KEYS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_api_keys(keys):
    with open(KEYS_FILE, 'w') as f:
        json.dump(keys, f, indent=2)

def load_fees():
    if os.path.exists(FEES_FILE):
        try:
            with open(FEES_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_fees(fees):
    with open(FEES_FILE, 'w') as f:
        json.dump(fees, f, indent=2)

DEFAULT_FEES = {
    'binance': 0.1, 'bybit': 0.1, 'okx': 0.1, 'gate': 0.2, 'kucoin': 0.1,
    'bitget': 0.1, 'hyperliquid': 0.05, 'bitmex': 0.075, 'bingx': 0.1,
    'htx': 0.2, 'mexc': 0.2, 'bitmart': 0.25, 'cryptocom': 0.1,
    'coinex': 0.1, 'woo': 0.02, 'bitfinex': 0.1, 'kraken': 0.16,
    'gemini': 0.1, 'phemex': 0.1, 'whitebit': 0.1,
}

EXCHANGES = list(DEFAULT_FEES.keys())

# ---------- Модель стакана ----------
class OrderBook:
    def __init__(self, p0_ask, k_ask, p0_bid, k_bid, max_ask_vol, max_bid_vol, fee=0.001):
        self.p0_ask = p0_ask
        self.k_ask = k_ask
        self.p0_bid = p0_bid
        self.k_bid = k_bid
        self.max_ask_volume = max_ask_vol
        self.max_bid_volume = max_bid_vol
        self.fee = fee

    def get_avg_price(self, side, volume):
        if volume <= 0:
            return self.p0_ask if side == 'ask' else self.p0_bid
        if side == 'ask':
            return self.p0_ask + self.k_ask * volume / 2.0
        else:
            return self.p0_bid - self.k_bid * volume / 2.0

    def exchange(self, from_volume, side):
        if side == 'buy':
            if from_volume > self.max_ask_volume:
                return 0.0
            avg_price = self.get_avg_price('ask', from_volume)
            received = from_volume / avg_price
        else:
            if from_volume > self.max_bid_volume:
                return 0.0
            avg_price = self.get_avg_price('bid', from_volume)
            received = from_volume * avg_price
        received *= (1 - self.fee)
        return received

# ---------- Класс биржи ----------
class RealExchange:
    def __init__(self, exchange_name, api_key='', secret='', fee=0.001):
        self.exchange = getattr(ccxt, exchange_name)({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
        })
        self.pairs = {}
        self.fee = fee

    def load_markets(self):
        self.exchange.load_markets()

    def fetch_orderbook(self, symbol, limit=20):
        return self.exchange.fetch_orderbook(symbol, limit=limit)

    def build_orderbooks(self, base_assets, quote_assets=None):
        if quote_assets is None:
            quote_assets = base_assets
        all_symbols = self.exchange.symbols
        for symbol in all_symbols:
            if '/' not in symbol:
                continue
            base, quote = symbol.split('/')
            if base in base_assets and quote in quote_assets:
                try:
                    ob_data = self.fetch_orderbook(symbol, limit=20)
                    if not ob_data['asks'] or not ob_data['bids']:
                        continue
                    asks = np.array(ob_data['asks'])
                    bids = np.array(ob_data['bids'])

                    p0_ask = asks[0, 0]
                    cum_vol_ask = np.cumsum(asks[:, 1])
                    if len(cum_vol_ask) > 1:
                        x = cum_vol_ask[:min(10, len(cum_vol_ask))]
                        y = asks[:len(x), 0]
                        if len(x) > 1:
                            slope, _ = np.polyfit(x, y, 1)
                            k_ask = max(slope, 0.0)
                        else: k_ask = 0.0
                    else: k_ask = 0.0

                    max_ask_volume = float(np.sum(asks[:, 1]))

                    p0_bid = bids[0, 0]
                    cum_vol_bid = np.cumsum(bids[:, 1])
                    if len(cum_vol_bid) > 1:
                        x = cum_vol_bid[:min(10, len(cum_vol_bid))]
                        y = bids[:len(x), 0]
                        if len(x) > 1:
                            slope, _ = np.polyfit(x, y, 1)
                            k_bid = max(-slope, 0.0)
                        else: k_bid = 0.0
                    else: k_bid = 0.0

                    max_bid_volume = float(np.sum(bids[:, 1]))

                    self.pairs[(base, quote)] = OrderBook(
                        p0_ask, k_ask, p0_bid, k_bid,
                        max_ask_volume, max_bid_volume, self.fee
                    )
                except Exception:
                    continue

    def get_orderbook(self, base, quote):
        return self.pairs.get((base, quote))

    def get_rate(self, base, quote, volume_base=None):
        ob = self.get_orderbook(base, quote)
        if ob is None:
            return None
        if volume_base is None:
            return (ob.p0_ask + ob.p0_bid) / 2 * (1 - self.fee)
        else:
            return ob.exchange(volume_base, 'sell') / volume_base

    def execute_cycle(self, cycle, start_volume, log_callback):
        volume = start_volume
        log_callback(f"Начинаем исполнение цикла {' -> '.join(cycle)}\n")
        log_callback(f"Начальный объём: {volume:.4f} {cycle[0]}\n")
        for i in range(len(cycle)):
            from_asset = cycle[i]
            to_asset = cycle[(i+1) % len(cycle)]
            symbol = f"{to_asset}/{from_asset}"
            if symbol not in self.exchange.symbols:
                log_callback(f"Ошибка: пара {symbol} не найдена на бирже.\n")
                return False, 0, f"Пара {symbol} не найдена"
            try:
                if i == 0:
                    log_callback(f"Шаг {i+1}: Покупаем {to_asset} за {from_asset} на сумму {volume:.4f} {from_asset}\n")
                    # Используем create_order с параметром cost для универсальности
                    order = self.exchange.create_order(symbol, 'market', 'buy', None, None, {'cost': volume})
                    filled = order['filled'] if order['filled'] else 0
                    if filled == 0 and order.get('cost') and order.get('price'):
                        filled = order['cost'] / order['price']
                    log_callback(f" Получено {filled:.8f} {to_asset}\n")
                    volume = filled
                else:
                    log_callback(f"Шаг {i+1}: Продаём {from_asset} в количестве {volume:.8f} за {to_asset}\n")
                    order = self.exchange.create_market_sell_order(symbol, volume)
                    received = order['cost'] if order.get('cost') else 0
                    if received == 0 and order.get('filled') and order.get('price'):
                        received = order['filled'] * order['price']
                    log_callback(f" Получено {received:.4f} {to_asset}\n")
                    volume = received
                time.sleep(0.5)
            except Exception as e:
                log_callback(f"Ошибка при исполнении ордера: {str(e)}\n")
                return False, 0, f"Ошибка: {str(e)}"
        log_callback(f"Итоговая сумма: {volume:.4f} {cycle[0]}\n")
        return True, volume, "Успешно"

# ---------- Алгоритмы Беллмана-Форда ----------
def bellman_ford(weight, assets, max_cycles=10):
    n = len(weight)
    dist = [0] * n
    parent = [-1] * n
    for _ in range(n):
        updated = False
        for u in range(n):
            for v in range(n):
                if weight[u][v] < np.inf:
                    if dist[u] + weight[u][v] < dist[v] - 1e-12:
                        dist[v] = dist[u] + weight[u][v]
                        parent[v] = u
                        updated = True
        if not updated:
            break
    cycles = []
    visited_vertices = set()
    for u in range(n):
        for v in range(n):
            if weight[u][v] < np.inf and dist[u] + weight[u][v] < dist[v] - 1e-12:
                if v in visited_vertices: continue
                # Восстановление цикла
                cycle = []
                cur = v
                seen = {}
                while cur not in seen:
                    # Если parent[cur] == -1, цикл не найден – выходим
                    if parent[cur] == -1:
                        break
                    seen[cur] = len(cycle)
                    cycle.append(cur)
                    cur = parent[cur]
                else:
                    # Цикл найден
                    if cur in seen:
                        start_idx = seen[cur]
                        cycle = cycle[start_idx:]
                        if len(cycle) >= 2:
                            cycle_names = [assets[i] for i in cycle]
                            total_weight = sum(weight[cycle[i]][cycle[(i+1)%len(cycle)]] for i in range(len(cycle)))
                            if total_weight < -1e-12:
                                cycles.append(cycle_names)
                                for node in cycle:
                                    visited_vertices.add(node)
                                if len(cycles) >= max_cycles:
                                    return cycles
    return cycles

def profit_for_cycle(exchange, cycle, start_volume):
    volume = start_volume
    for i in range(len(cycle)):
        from_asset = cycle[i]
        to_asset = cycle[(i+1) % len(cycle)]
        ob = exchange.get_orderbook(from_asset, to_asset)
        if ob is None:
            return 0.0
        volume = ob.exchange(volume, 'sell')
        if volume <= 0:
            return 0.0
    return volume

def profit_function(exchange, cycle):
    def f(V):
        if V <= 0:
            return 0.0
        return profit_for_cycle(exchange, cycle, V) - V
    return f

def compute_max_start_volume(exchange, cycle, low=0.1, high=1e6, tol=1e-3):
    def can_cycle(V):
        vol = V
        for i in range(len(cycle)):
            from_asset = cycle[i]
            to_asset = cycle[(i+1) % len(cycle)]
            ob = exchange.get_orderbook(from_asset, to_asset)
            if ob is None:
                return False
            # Проверяем, что можем продать текущий объём
            if vol > ob.max_bid_volume:
                return False
            # Рассчитываем получаемый объём после продажи с учётом комиссии
            avg_price = ob.get_avg_price('bid', vol)
            vol = vol * avg_price * (1 - ob.fee)
            if vol <= 0:
                return False
        return True

    # Бинарный поиск максимального начального объёма
    hi = high
    if not can_cycle(hi):
        while hi > low and not can_cycle(hi):
            hi /= 2
        if hi <= low:
            return low

    lo = low
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if can_cycle(mid):
            lo = mid
        else:
            hi = mid
    return lo

def find_optimal_volume(exchange, cycle, low=1e-6, high=None, tol=1e-6):
    if high is None:
        high = compute_max_start_volume(exchange, cycle)
    if high <= low:
        return low, 0.0

    f = profit_function(exchange, cycle)
    res = minimize_scalar(lambda x: -f(x), bounds=(low, high), method='bounded', options={'xatol': tol})

    if res.success:
        return res.x, -res.fun
    else:
        best_v = low
        best_p = f(low)
        for v in np.linspace(low, high, 50):
            p = f(v)
            if p > best_p:
                best_p = p
                best_v = v
        return best_v, best_p
# ---------- ИНТЕРФЕЙС STREAMLIT ----------
st.title("Arbitrage Crypto Bot ⚡")

# Загрузка ключей
keys = load_api_keys()
exchange_name = st.selectbox("Выберите биржу", EXCHANGES)
api_key = st.text_input("API Key", value=keys.get(exchange_name, {}).get('api_key', ''), type="password")
secret = st.text_input("Secret Key", value=keys.get(exchange_name, {}).get('secret', ''), type="password")

if st.button("Сохранить ключи"):
    keys[exchange_name] = {'api_key': api_key, 'secret': secret}
    save_api_keys(keys)
    st.success("Ключи сохранены")

# Загрузка комиссий
fees = load_fees()
fee = fees.get(exchange_name, DEFAULT_FEES.get(exchange_name, 0.1)) / 100.0

# Параметры поиска
assets_input = st.text_input("Активы (через запятую)", "BTC, ETH, USDT")
base_assets = [x.strip() for x in assets_input.split(',') if x.strip()]

if st.button("Найти арбитражные циклы"):
    if not api_key or not secret:
        st.error("Введите API ключи")
    else:
        with st.spinner("Загрузка данных..."):
            ex = RealExchange(exchange_name, api_key, secret, fee)
            ex.load_markets()
            ex.build_orderbooks(base_assets, base_assets)
            
            # Построение матрицы весов
            n = len(base_assets)
            weight = np.full((n, n), np.inf)
            for i, base in enumerate(base_assets):
                for j, quote in enumerate(base_assets):
                    if i != j:
                        rate = ex.get_rate(base, quote)
                        if rate is not None and rate > 0:
                            weight[i][j] = -math.log(rate)
            
            cycles = bellman_ford(weight, base_assets)
            if cycles:
                st.success(f"Найдено {len(cycles)} циклов:")
                for cycle in cycles:
                    st.write(" → ".join(cycle))
                    # Рассчёт оптимального объёма
                    opt_v, profit = find_optimal_volume(ex, cycle)
                    st.write(f"  Оптимальный стартовый объём: {opt_v:.4f} {cycle[0]}, прибыль: {profit:.4f} {cycle[0]}")
            else:
                st.warning("Циклов не найдено")
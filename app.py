import math
import numpy as np
import ccxt
import streamlit as st
from scipy.optimize import minimize_scalar
import time
import json
import os
from datetime import datetime

st.set_page_config(page_title="Arbitrage Crypto Bot", page_icon="⚡", layout="centered")

# ---------- Файлы настроек ----------
KEYS_FILE = "api_keys.json"
FEES_FILE = "fees.json"

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

# ---------- Класс биржи (расширенный) ----------
class RealExchange:
    def __init__(self, exchange_name, api_key='', secret='', fee=0.001):
        self.exchange = getattr(ccxt, exchange_name)({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
        })
        self.fee = fee
        self.spot_pairs = {}      # (base, quote) -> OrderBook (спот)
        self.swap_pairs = {}      # (base, quote) -> OrderBook (фьючерсы: бессрочные и срочные)
        self.future_markets = {}  # symbol -> market info (только фьючерсные рынки)

    def load_markets(self):
        self.exchange.load_markets()
        for symbol, market in self.exchange.markets.items():
            if market['type'] == 'spot':
                self.spot_pairs[symbol] = market
            elif market['type'] in ['swap', 'future'] and market.get('linear'):
                self.future_markets[symbol] = market

    def fetch_orderbook(self, symbol, limit=20):
        return self.exchange.fetch_orderbook(symbol, limit=limit)

    def build_orderbooks(self, base_assets, quote_assets=None):
        if quote_assets is None:
            quote_assets = base_assets
        self.spot_obs = {}
        for symbol in self.exchange.symbols:
            if '/' not in symbol:
                continue
            base, quote = symbol.split('/')
            if base in base_assets and quote in quote_assets:
                market = self.exchange.markets.get(symbol)
                if market and market['type'] == 'spot':
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
                            slope, _ = np.polyfit(x, y, 1)
                            k_ask = max(slope, 0.0)
                        else: k_ask = 0.0
                        max_ask_volume = float(np.sum(asks[:, 1]))
                        p0_bid = bids[0, 0]
                        cum_vol_bid = np.cumsum(bids[:, 1])
                        if len(cum_vol_bid) > 1:
                            x = cum_vol_bid[:min(10, len(cum_vol_bid))]
                            y = bids[:len(x), 0]
                            slope, _ = np.polyfit(x, y, 1)
                            k_bid = max(-slope, 0.0)
                        else: k_bid = 0.0
                        max_bid_volume = float(np.sum(bids[:, 1]))
                        self.spot_obs[(base, quote)] = OrderBook(
                            p0_ask, k_ask, p0_bid, k_bid,
                            max_ask_volume, max_bid_volume, self.fee
                        )
                    except Exception:
                        continue

    def build_swap_orderbooks(self, base_assets, quote_assets=None):
        if quote_assets is None:
            quote_assets = base_assets
        self.swap_obs = {}
        for symbol, market in self.future_markets.items():
            base = market['base']
            quote = market['quote']
            if base in base_assets and quote in quote_assets:
                try:
                    ob_data = self.fetch_orderbook(symbol, limit=20)
                    if not ob_data['asks'] or not ob_data['bids']:
                        continue
                    asks = np.array(ob_data['asks'])
                    bids = np.array(ob_data['bids'])
                    p0_ask = asks[0, 0]
                    cum_vol_ask = np.cumsum(asks[:, 1])
                    k_ask = 0.0
                    if len(cum_vol_ask) > 1:
                        x = cum_vol_ask[:min(10, len(cum_vol_ask))]
                        y = asks[:len(x), 0]
                        slope, _ = np.polyfit(x, y, 1)
                        k_ask = max(slope, 0.0)
                    max_ask_volume = float(np.sum(asks[:, 1]))
                    p0_bid = bids[0, 0]
                    cum_vol_bid = np.cumsum(bids[:, 1])
                    k_bid = 0.0
                    if len(cum_vol_bid) > 1:
                        x = cum_vol_bid[:min(10, len(cum_vol_bid))]
                        y = bids[:len(x), 0]
                        slope, _ = np.polyfit(x, y, 1)
                        k_bid = max(-slope, 0.0)
                    max_bid_volume = float(np.sum(bids[:, 1]))
                    # Фьючерсная комиссия может отличаться, берём из маркета или общую
                    fee_swap = self.fee  # можно переопределить через market['taker'] и т.д.
                    self.swap_obs[(base, quote)] = OrderBook(
                        p0_ask, k_ask, p0_bid, k_bid,
                        max_ask_volume, max_bid_volume, fee_swap
                    )
                except Exception:
                    continue

    def get_spot_orderbook(self, base, quote):
        return self.spot_obs.get((base, quote))

    def get_swap_orderbook(self, base, quote):
        return self.swap_obs.get((base, quote))

    def get_rate(self, base, quote, volume_base=None):
        ob = self.get_spot_orderbook(base, quote)
        if ob is None:
            return None
        if volume_base is None:
            return (ob.p0_ask + ob.p0_bid) / 2 * (1 - self.fee)
        else:
            return ob.exchange(volume_base, 'sell') / volume_base

    # Методы исполнения циклов (оставлены без изменений для спота)
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

# ---------- Функции треугольного арбитража ----------
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
                if v in visited_vertices:
                    continue
                cycle = []
                cur = v
                seen = {}
                while cur not in seen:
                    if parent[cur] == -1:
                        break
                    seen[cur] = len(cycle)
                    cycle.append(cur)
                    cur = parent[cur]
                else:
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
        ob = exchange.get_spot_orderbook(from_asset, to_asset)
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
            ob = exchange.get_spot_orderbook(from_asset, to_asset)
            if ob is None:
                return False
            if vol > ob.max_bid_volume:
                return False
            avg_price = ob.get_avg_price('bid', vol)
            vol = vol * avg_price * (1 - ob.fee)
            if vol <= 0:
                return False
        return True

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

# ---------- Новые функции для арбитража спот-фьючерс ----------
def calculate_cash_and_carry(exchange, base, quote_asset, spot_ob, swap_ob, swap_market):
    """
    Рассчитывает прибыль от операции cash-and-carry (или reverse) для пары base/quote.
    Возвращает словарь с параметрами или None, если нет возможности.
    """
    # Определяем направление: если фьючерс выше спота с учётом комиссий -> cash-and-carry (покупка спот, продажа фьюча)
    # иначе reverse.
    spot_bid = spot_ob.p0_bid
    spot_ask = spot_ob.p0_ask
    swap_bid = swap_ob.p0_bid
    swap_ask = swap_ob.p0_ask

    fee_spot = spot_ob.fee
    fee_swap = swap_ob.fee

    # Стоимость покупки на споте (используем ask) и продажи на фьючерсе (bid)
    cost_buy_spot = spot_ask * (1 + fee_spot)  # цена с учётом комиссии тейкера
    revenue_sell_swap = swap_bid * (1 - fee_swap)

    # Стоимость продажи на споте (bid) и покупки на фьючерсе (ask)
    revenue_sell_spot = spot_bid * (1 - fee_spot)
    cost_buy_swap = swap_ask * (1 + fee_swap)

    # funding rate (если swap)
    funding_rate = 0.0
    if swap_market['type'] == 'swap':
        try:
            fr_data = exchange.exchange.fetch_funding_rate(swap_market['symbol'])
            funding_rate = fr_data['fundingRate'] if fr_data else 0.0
        except:
            pass

    # Время до экспирации (в днях)
    days_to_expiry = 365  # для бессрочных условно год, для срочных считаем
    if swap_market.get('expiry'):
        expiry_dt = datetime.fromtimestamp(swap_market['expiry'] / 1000)
        now = datetime.now()
        days_to_expiry = max((expiry_dt - now).days, 1)

    # Расчёт для cash-and-carry (long spot, short future)
    # Прибыль на 1 контракт (1 единицу базового актива):
    # Сейчас покупаем base за quote на споте: тратим cost_buy_spot (в quote)
    # Продаём фьючерс: получаем revenue_sell_swap (в quote) при экспирации (или сейчас, если бессрочный? Упростим: считаем разницу цен).
    # Дополнительно: для бессрочных нужно учесть funding rate, который мы будем платить/получать за период удержания.
    # Предполагаем, что позиция держится до экспирации (или бесконечно для swap, но мы упрощаем до 1 дня? Нет, можно дать годовую доходность, если держим вечно, но это спекулятивно).
    # Более консервативно: для swap считаем прибыль за 1 период финансирования (обычно 8 часов), но доходность в годовом выражении APR.

    # Определим, есть ли арбитраж:
    # Вариант 1: cash-and-carry: cost_buy_spot < revenue_sell_swap ?
    # Но на споте мы покупаем, на фьючерсе продаём - обе сделки совершаются сейчас по текущим ценам.
    # Однако прибыль получается мгновенно? Нет: при cash-and-carry мы покупаем базовый актив и одновременно продаём фьючерс,
    # фиксируя цену продажи. К экспирации фьючерс сойдётся с ценой спота, прибыль = (цена продажи - цена покупки) - комиссии.
    # Поэтому достаточно сравнить эффективную цену продажи (с учётом комиссий) и цену покупки.
    # Для безрискового арбитража нужно: spot_ask * (1+fee_spot) < swap_bid * (1-fee_swap) (если цена фьючерса выше спота).
    # Обратное для reverse.

    # Определим лимиты по объёму: максимум, который можно исполнить на споте и на фьючерсе одновременно.
    # Объём в базовом активе.
    max_vol_spot_ask = spot_ob.max_ask_volume  # можем купить на споте столько base
    max_vol_swap_bid = swap_ob.max_bid_volume  # можем продать на фьючерсе столько контрактов

    results = []

    # Проверяем cash-and-carry (покупка спот, продажа фьючерса)
    if cost_buy_spot < revenue_sell_swap:
        # Можем исполнить объём, ограниченный минимальным из двух
        vol_limit = min(max_vol_spot_ask, max_vol_swap_bid)
        if vol_limit <= 0:
            return None
        # Оптимальный объём: можно искать максимум прибыли, но простой подход - взять vol_limit,
        # так как доход линейный (если стаканы не сильно наклонные). Для точности можно учесть проскальзывание.
        # Построим функцию прибыли от объёма V:
        # Покупаем V base на споте: потратим sum(ask_prices) с учётом комиссии.
        # Используем модель OrderBook.exchange: spot_ob.exchange(V * spot_ask?, не совсем.
        # exchange(from_volume, 'buy') ожидает объём в quote? В нашей модели обмен несимметричный.
        # Для спота: покупка base за quote: передаём объём в quote (сумму, которую тратим), получаем объём base.
        # Но здесь мы фиксируем объём base, который хотим купить. Поэтому нужно адаптировать.
        # Вместо этого быстро вычислим среднюю цену покупки для V_base:
        def spot_buy_cost(v_base):
            if v_base <= 0: return 0
            # Используем стакан: проходим по уровням, пока не наберём v_base.
            # У нас нет прямого метода, поэтому аппроксимируем: цена ask при объёме v_base равна p0_ask + k_ask * v_base (интеграл).
            # Тогда средняя цена покупки = p0_ask + k_ask * v_base / 2
            avg_price = spot_ob.get_avg_price('ask', v_base)
            return v_base * avg_price * (1 + fee_spot)

        def swap_sell_revenue(v_base):
            if v_base <= 0: return 0
            avg_price = swap_ob.get_avg_price('bid', v_base)
            return v_base * avg_price * (1 - fee_swap)

        profit = lambda v: swap_sell_revenue(v) - spot_buy_cost(v)
        # Найдём максимум profit на [0, vol_limit]
        opt_vol = vol_limit
        if vol_limit > 0.1:
            res = minimize_scalar(lambda v: -profit(v), bounds=(0.1, vol_limit), method='bounded', options={'xatol': 0.001})
            if res.success:
                opt_vol = res.x
        max_profit = profit(opt_vol)
        if max_profit > 0:
            # APR: (прибыль / затраты) * (365 / days_to_expiry)
            cost = spot_buy_cost(opt_vol)
            apr = (max_profit / cost) * (365 / days_to_expiry) * 100 if cost > 0 else 0
            results.append({
                'type': 'cash-and-carry',
                'base': base,
                'quote': quote_asset,
                'direction': f'Купить {base} на споте, продать на фьючерсе',
                'volume_base': opt_vol,
                'profit': max_profit,
                'cost': cost,
                'apr': apr,
                'funding_rate': funding_rate,
                'days_to_expiry': days_to_expiry
            })

    # Проверяем reverse cash-and-carry (продажа спот, покупка фьючерса)
    if revenue_sell_spot > cost_buy_swap:
        max_vol_spot_bid = spot_ob.max_bid_volume
        max_vol_swap_ask = swap_ob.max_ask_volume
        vol_limit = min(max_vol_spot_bid, max_vol_swap_ask)
        if vol_limit <= 0:
            return None

        def spot_sell_revenue(v_base):
            if v_base <= 0: return 0
            avg_price = spot_ob.get_avg_price('bid', v_base)
            return v_base * avg_price * (1 - fee_spot)

        def swap_buy_cost(v_base):
            if v_base <= 0: return 0
            avg_price = swap_ob.get_avg_price('ask', v_base)
            return v_base * avg_price * (1 + fee_swap)

        profit = lambda v: spot_sell_revenue(v) - swap_buy_cost(v)
        opt_vol = vol_limit
        if vol_limit > 0.1:
            res = minimize_scalar(lambda v: -profit(v), bounds=(0.1, vol_limit), method='bounded', options={'xatol': 0.001})
            if res.success:
                opt_vol = res.x
        max_profit = profit(opt_vol)
        if max_profit > 0:
            cost = swap_buy_cost(opt_vol)
            apr = (max_profit / cost) * (365 / days_to_expiry) * 100 if cost > 0 else 0
            results.append({
                'type': 'reverse cash-and-carry',
                'base': base,
                'quote': quote_asset,
                'direction': f'Продать {base} на споте, купить на фьючерсе',
                'volume_base': opt_vol,
                'profit': max_profit,
                'cost': cost,
                'apr': apr,
                'funding_rate': funding_rate,
                'days_to_expiry': days_to_expiry
            })

    return results if results else None

def find_all_cash_and_carry(exchange, base_assets, quote_assets):
    opportunities = []
    for base in base_assets:
        for quote in quote_assets:
            spot_ob = exchange.get_spot_orderbook(base, quote)
            swap_ob = exchange.get_swap_orderbook(base, quote)
            if spot_ob and swap_ob:
                # Ищем соответствующий swap_market
                swap_market = None
                for sym, mkt in exchange.future_markets.items():
                    if mkt['base'] == base and mkt['quote'] == quote:
                        swap_market = mkt
                        break
                if swap_market:
                    res = calculate_cash_and_carry(exchange, base, quote, spot_ob, swap_ob, swap_market)
                    if res:
                        opportunities.extend(res)
    # Сортируем по APR
    opportunities.sort(key=lambda x: x['apr'], reverse=True)
    return opportunities

# ---------- Streamlit Interface ----------
st.title("Arbitrage Crypto Bot ⚡")

keys = load_api_keys()
exchange_name = st.selectbox("Выберите биржу", EXCHANGES)
api_key = st.text_input("API Key", value=keys.get(exchange_name, {}).get('api_key', ''), type="password")
secret = st.text_input("Secret Key", value=keys.get(exchange_name, {}).get('secret', ''), type="password")

if st.button("Сохранить ключи"):
    keys[exchange_name] = {'api_key': api_key, 'secret': secret}
    save_api_keys(keys)
    st.success("Ключи сохранены")

fees = load_fees()
fee = fees.get(exchange_name, DEFAULT_FEES.get(exchange_name, 0.1)) / 100.0

mode = st.radio("Режим поиска", ["Треугольный арбитраж (спот)", "Арбитраж спот-фьючерс"])

assets_input = st.text_input("Активы (через запятую)", "BTC, ETH, USDT")
base_assets = [x.strip() for x in assets_input.split(',') if x.strip()]

if st.button("Найти арбитраж"):
    if not api_key or not secret:
        st.error("Введите API ключи")
    else:
        with st.spinner("Загрузка данных..."):
            ex = RealExchange(exchange_name, api_key, secret, fee)
            ex.load_markets()

            if mode == "Треугольный арбитраж (спот)":
                ex.build_orderbooks(base_assets, base_assets)
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
                        opt_v, profit = find_optimal_volume(ex, cycle)
                        st.write(f"  Оптимальный стартовый объём: {opt_v:.4f} {cycle[0]}, прибыль: {profit:.4f} {cycle[0]}")
                else:
                    st.warning("Циклов не найдено")

            else:  # Арбитраж спот-фьючерс
                # Строим спотовые и фьючерсные стаканы
                quote_assets = base_assets  # обычно USDT
                ex.build_orderbooks(base_assets, quote_assets)
                ex.build_swap_orderbooks(base_assets, quote_assets)
                opps = find_all_cash_and_carry(ex, base_assets, quote_assets)
                if opps:
                    st.success(f"Найдено {len(opps)} возможностей спот-фьючерс:")
                    for op in opps:
                        st.write(f"**{op['base']}/{op['quote']}** — {op['type']}")
                        st.write(f"  {op['direction']}")
                        st.write(f"  Оптимальный объём (базовый актив): {op['volume_base']:.4f} {op['base']}")
                        st.write(f"  Ожидаемая прибыль: {op['profit']:.2f} {op['quote']}")
                        st.write(f"  Годовая доходность (APR): {op['apr']:.2f}%")
                        if op['funding_rate']:
                            st.write(f"  Текущая ставка финансирования: {op['funding_rate']*100:.4f}%")
                        st.write(f"  Дней до экспирации (если срочный): {op['days_to_expiry']} дней")
                        st.write("---")
                else:
                    st.warning("Возможностей спот-фьючерс не найдено")
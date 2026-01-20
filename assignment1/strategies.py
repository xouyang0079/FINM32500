from abc import ABC, abstractmethod
from models import MarketDataPoint
from typing import Optional


class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, tick: MarketDataPoint) -> list:
        pass

class MAC(Strategy):
    def __init__(self, short_win, long_win, qty: Optional = None):
        self.short_win = short_win
        self.long_win = long_win
        self.qty = qty

        # rolling price buffer
        self._prices = []

    def generate_signals(self, tick: MarketDataPoint) -> list:
        self._prices.append(tick.price)

        # keep buffer size upper bound
        if len(self._prices) > self.long_win + 1:
            self._prices.pop(0)

        # keep bufffer size lower bound
        if len(self._prices) < self.long_win + 1:
            return []
        
        s_t_1 = self._prices[:-1][-self.short_win:]
        sma_s_t_1 = sum(s_t_1) / len(s_t_1)

        s_t = self._prices[-self.short_win:]
        sma_s_t = sum(s_t) / len(s_t)

        l_t_1 = self._prices[:-1][-self.long_win:]
        sma_l_t_1 = sum(l_t_1) / len(l_t_1)

        l_t = self._prices[-self.long_win:]
        sma_l_t = sum(l_t) / len(l_t)

        if sma_s_t_1 <= sma_l_t_1 and sma_s_t > sma_l_t:
            return [('BUY', tick.symbol, self.qty, tick.price)]
        
        if sma_s_t_1 >= sma_l_t_1 and sma_s_t < sma_l_t:
            return [('SELL', tick.symbol, self.qty, tick.price)]

        return []


class Momentum(Strategy):
    def __init__(self, lookback, qty: Optional = None):
        self.lookback = lookback
        self.qty = qty

        self._prices = []
    
    def generate_signals(self, tick: MarketDataPoint) -> list:
        self._prices.append(tick.price)

        if len(self._prices) > self.lookback + 1:
            self._prices.pop(0)
        
        if len(self._prices) < self.lookback + 1:
            return []

        if self._prices[-1] > self._prices[-1-self.lookback]:
            return [('BUY', tick.symbol, self.qty, tick.price)]
        elif self._prices[-1] < self._prices[-1-self.lookback]:
            return [('SELL', tick.symbol, self.qty, tick.price)]
        else:
            return []
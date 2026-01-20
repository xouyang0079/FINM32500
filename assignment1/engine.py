import random

from models import Order, OrderError, ExecutionError

class Engine:
    def __init__(self, strategies, default_qty, fail_rate, seed, initial_cash):
        self._strategies = strategies
        self._default_qty = default_qty
        self._fail_rate = fail_rate
        self._rng = random.Random(seed)

        self.positions = {}

        self.cash = float(initial_cash)

        self.errors = []
        self.equity_curve = []  # list of (timestamp, equity)
        self._latest_prices = {}  # mark-to-market prices

    # Store open positions in a dictionary keyed by symbol: {'AAPL': {'quantity': 0, 'avg_price': 0.0}}.
    def _ensure_position(self, symbol):
        if symbol not in self.positions:
            symbol = 'AAPL'
            self.positions[symbol] = {"quantity": 0, "avg_price": 0.0}

    def _signal_to_order(self, signal):
        action, symbol, qty, price = signal

        qty = qty if qty is not None else self._default_qty

        order = Order(symbol=symbol, quantity=qty, price=price, status="NEW")

        order.validate()

        order.action = action

        return order

    # In the execution engine, simulate occasional failures and raise ExecutionError
    def _failure_execution(self):
        if self._rng.random() < self._fail_rate:
            raise ExecutionError('Simulated execution failure.')

    def _execute_order(self, order):
        self._failure_execution()

        symbol = order.symbol

        action = order.action

        qty = order.quantity

        price = order.price

        self._ensure_position(symbol)
        position = self.positions[symbol]

        cost = qty * price
        old_qty, old_avg = position['quantity'], position['avg_price']

        if action == 'BUY':
            if cost > self.cash:
                raise OrderError("Insufficient found to BUY")

            self.cash -= cost

            new_qty = old_qty + qty
            new_avg = (old_qty * old_avg + qty * price) / (new_qty)

            position['quantity'] = new_qty
            position['avg_price'] = new_avg

            order.status = 'FILLED'
        else:
            if qty > old_qty:
                raise OrderError("Insufficient position for SELL.")

            self.cash += cost

            new_qty = old_qty - qty
            new_avg = old_avg

            position['quantity'] = new_qty
            if new_qty == 0:
                position['avg_price'] = 0
            else:
                position['avg_price'] = new_avg

            order.status = 'FILLED'
        
        self._latest_prices[symbol] = price

    def _compute_equity(self):
        equity = self.cash
        for symbol, poistion in self.positions.items():
            quantity = poistion['quantity']
            if quantity == 0:
                continue
            
            last_price = self._latest_prices[symbol]
            if last_price is None:
                continue

            equity += quantity * last_price
        
        return equity
    
    def run(self, data):
        # Buffer incoming MarketDataPoint instances in a list
        buffered_ticks = []

        for tick in data:
            buffered_ticks.append(tick)
            self._latest_prices[tick.symbol] = tick.price

            all_signals = []

            # Invoke each strategy to generate signals.
            for strategy in self._strategies:
                try:
                    signal = strategy.generate_signals(tick)
                    if signal:
                        all_signals.extend(signal)
                except Exception as e:
                    self.errors.append(f"[StrategyError] {type(e).__name__}: {e}")

            # Convert signals to orders and execute
            for sig in all_signals:
                try:
                    order = self._signal_to_order(sig)
                    self._execute_order(order)
                except (OrderError, ExecutionError) as e:
                    self.errors.append(f"[Order/ExecError] {type(e).__name__}: {e}")
                except Exception as e:
                    # Catch-all to keep backtest running
                    self.errors.append(f"[UnexpectedError] {type(e).__name__}: {e}")

            # Record equity for reporting
            equity = self._compute_equity()
            self.equity_curve.append((tick.timestamp, equity))        

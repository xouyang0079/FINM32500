import argparse
import csv
import os
import random
import time

from naive_order import NaiveOrderBook
from optimized_order import OptimizedOrderBook


# Generate benchmark orders
class OrderGenerator:
    def gen_orders(self, n, seed=42):
        rng = random.Random(seed)
        orders = []
        for i in range(1, n + 1):
            side = "bid" if (i % 2 == 0) else "ask"
            price = rng.uniform(50.0, 200.0)
            qty = rng.randint(1, 100)
            orders.append({"order_id": i, "price": price, "quantity": qty, "side": side})
        return orders


# Measure time
class Timer:
    def timeit(self, fn):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        return t1 - t0


# Save results
class CsvWriter:
    def save(self, rows, path):
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)


# Choose order book
class OrderBookFactory:
    def create(self, mode):
        if mode == "naive":
            return NaiveOrderBook
        if mode == "optimized":
            return OptimizedOrderBook
        raise ValueError(f"Invalid mode: {mode}")


# Benchmark book operations
class BenchmarkRunner:
    def __init__(self, book_cls, orders, timer):
        self.book_cls = book_cls
        self.orders = orders
        self.timer = timer

    # Benchmark insert
    def benchmark_insert(self, n):
        book = self.book_cls()

        def run():
            for i in range(n):
                book.add_order(self.orders[i])

        total = self.timer.timeit(run)
        avg = total / n if n > 0 else float("nan")
        return total, avg

    # Benchmark amend
    def benchmark_amend(self, n, seed=123):
        # before amending order, we need to add order first
        book = self.book_cls()
        for i in range(n):
            book.add_order(self.orders[i])

        rng = random.Random(seed)
        order_ids = [self.orders[i]["order_id"] for i in range(n)]

        # amend order
        def run():
            for _ in range(n):
                oid = rng.choice(order_ids)
                new_qty = rng.randint(1, 100)
                book.amend_order(oid, new_qty)

        total = self.timer.timeit(run)
        avg = total / n if n > 0 else float("nan")
        return total, avg

    # Benchmark delete
    def benchmark_delete(self, n, seed=999):
        # before deleting order, we need to add order first
        book = self.book_cls()
        for i in range(n):
            book.add_order(self.orders[i])

        rng = random.Random(seed)
        order_ids = [self.orders[i]["order_id"] for i in range(n)]

        # the order id should be randomly shuffle
        rng.shuffle(order_ids)

        def run():
            for oid in order_ids:
                book.delete_order(oid)

        total = self.timer.timeit(run)
        avg = total / n if n > 0 else float("nan")
        return total, avg

    # Benchmark lookup_by_id
    def benchmark_lookup(self, n, seed=777):
        book = self.book_cls()
        for i in range(n):
            book.add_order(self.orders[i])

        rng = random.Random(seed)
        order_ids = [self.orders[i]["order_id"] for i in range(n)]

        def run():
            for _ in range(n):
                oid = rng.choice(order_ids)
                book.lookup_by_id(oid)

        total = self.timer.timeit(run)
        avg = total / n if n > 0 else float("nan")
        return total, avg

    # Benchmark get_orders_at_price
    def benchmark_retrieve_price(self, n, seed=888):
        book = self.book_cls()
        for i in range(n):
            book.add_order(self.orders[i])

        rng = random.Random(seed)

        # Sample prices that actually exist in the book (from the inserted orders)
        prices = [self.orders[i]["price"] for i in range(n)]
        sides = [None, "bid", "ask"]

        def run():
            for _ in range(n):
                p = rng.choice(prices)
                s = rng.choice(sides)
                book.get_orders_at_price(p, side=s)

        total = self.timer.timeit(run)
        avg = total / n if n > 0 else float("nan")
        return total, avg

    # Benchmark best_bid
    def benchmark_best_bid(self, n):
        book = self.book_cls()
        for i in range(n):
            book.add_order(self.orders[i])

        def run():
            for _ in range(n):
                book.best_bid()

        total = self.timer.timeit(run)
        avg = total / n if n > 0 else float("nan")
        return total, avg

    # Benchmark best_ask
    def benchmark_best_ask(self, n):
        book = self.book_cls()
        for i in range(n):
            book.add_order(self.orders[i])

        def run():
            for _ in range(n):
                book.best_ask()

        total = self.timer.timeit(run)
        avg = total / n if n > 0 else float("nan")
        return total, avg


# Benchmark
class Benchmark:
    def __init__(self, book_mode, include_queries=False):
        self.book_mode = book_mode
        self.include_queries = include_queries

        self.generator = OrderGenerator()
        self.timer = Timer()
        self.writer = CsvWriter()
        self.factory = OrderBookFactory()

    def run(self):
        ns = [10, 100, 1000, 10000, 100000, 1000000]
        orders = self.generator.gen_orders(max(ns), seed=42)

        book_cls = self.factory.create(self.book_mode)
        runner = BenchmarkRunner(book_cls=book_cls, orders=orders, timer=self.timer)

        rows = []
        insert_total = []
        amend_total = []
        delete_total = []

        lookup_total = []
        retrieve_total = []
        best_bid_total = []
        best_ask_total = []

        for n in ns:
            total, avg = runner.benchmark_insert(n)
            insert_total.append(total)
            rows.append({"method": self.book_mode, "operation": "insert", "n": n, "total_sec": total, "avg_sec": avg})

            total, avg = runner.benchmark_amend(n, seed=123)
            amend_total.append(total)
            rows.append({"method": self.book_mode, "operation": "amend", "n": n, "total_sec": total, "avg_sec": avg})

            total, avg = runner.benchmark_delete(n, seed=456)
            delete_total.append(total)
            rows.append({"method": self.book_mode, "operation": "delete", "n": n, "total_sec": total, "avg_sec": avg})

            print(
                f"[{self.book_mode} n={n}] "
                f"insert={insert_total[-1]:.6f}s | amend={amend_total[-1]:.6f}s | delete={delete_total[-1]:.6f}s"
            )

            if self.include_queries:
                total, avg = runner.benchmark_lookup(n, seed=777)
                lookup_total.append(total)
                rows.append({"method": self.book_mode, "operation": "lookup_by_id", "n": n, "total_sec": total, "avg_sec": avg})

                total, avg = runner.benchmark_retrieve_price(n, seed=888)
                retrieve_total.append(total)
                rows.append({"method": self.book_mode, "operation": "get_orders_at_price", "n": n, "total_sec": total, "avg_sec": avg})

                total, avg = runner.benchmark_best_bid(n)
                best_bid_total.append(total)
                rows.append({"method": self.book_mode, "operation": "best_bid", "n": n, "total_sec": total, "avg_sec": avg})

                total, avg = runner.benchmark_best_ask(n)
                best_ask_total.append(total)
                rows.append({"method": self.book_mode, "operation": "best_ask", "n": n, "total_sec": total, "avg_sec": avg})

                print(
                    f"    queries: lookup={lookup_total[-1]:.6f}s | "
                    f"retrieve={retrieve_total[-1]:.6f}s | "
                    f"best_bid={best_bid_total[-1]:.6f}s | best_ask={best_ask_total[-1]:.6f}s"
                )

        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_csv = os.path.join(base_dir, f"benchmark_results_{self.book_mode}.csv")
        self.writer.save(rows, results_csv)

        print(f"Results save to: {os.path.basename(results_csv)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--book",
        type=str,
        default="naive",
        choices=["naive", "optimized"],
        help="choose method: naive or optimized"
    )
    parser.add_argument(
        "--include-queries",
        type=bool,
        default="True",
        help="benchmark lookup_by_id / get_orders_at_price / best_bid / best_ask"
    )
    args = parser.parse_args()

    app = Benchmark(book_mode=args.book, include_queries=args.include_queries)
    app.run()


if __name__ == "__main__":
    main()

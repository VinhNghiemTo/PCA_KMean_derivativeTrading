from __future__ import (absolute_import, division, print_function, unicode_literals)

import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import rework_backtrader as rbt
from rework_backtrader.indicators import RSI, AverageTrueRange, SimpleMovingAverage, BollingerBands


class AdaptivePCAKMeansAlpha(rbt.rbtstrategies.RopylabStrategy):
    params = dict(
        atr_period=14,
        rsi_period=14,
        ma_fast_period=10,
        ma_slow_period=30,
        bb_period=20,
        pca_components=1,
        k_clusters=3,
        train_window=120,
        trade_fraction=1.0
    )

    def __init__(self):
        super(AdaptivePCAKMeansAlpha, self).__init__()
        self.order = None

        # Indicators
        self.atr = AverageTrueRange(self.data, period=self.params.atr_period)
        self.rsi = RSI(self.data.close, period=self.params.rsi_period)
        self.ma_fast = SimpleMovingAverage(self.data.close, period=self.params.ma_fast_period)
        self.ma_slow = SimpleMovingAverage(self.data.close, period=self.params.ma_slow_period)
        self.bb = BollingerBands(self.data.close, period=self.params.bb_period)

        # Models
        self.pca = PCA(n_components=self.params.pca_components)
        self.kmeans = KMeans(n_clusters=self.params.k_clusters, random_state=42)

    def get_third_thursday(self, date):
        """Get the third Thursday of the month."""
        first_day_of_month = date.replace(day=1)
        first_thursday = first_day_of_month + datetime.timedelta(days=(3 - first_day_of_month.weekday()) % 7)
        return first_thursday + datetime.timedelta(weeks=2)

    def find_best_short_timing(self):
        """Find the best timing to short all positions during the expiration week."""
        current_date = self.data.datetime.date(0)
        third_thursday = self.get_third_thursday(current_date)

        # Determine if it's the expiration week (Monday to Thursday)
        expiration_week_start = third_thursday - datetime.timedelta(days=4)
        expiration_week_end = third_thursday

        if expiration_week_start <= current_date <= expiration_week_end:
            upper_band = self.bb.top[0]
            rsi_value = self.rsi[0]
            current_price = self.data.close[0]

            self.log(
                f"Date: {current_date}, Price: {current_price:.2f}, "
                f"Upper Band: {upper_band:.2f}, RSI: {rsi_value:.2f}"
            )

            # Short if price near upper band or RSI > 70
            if current_price >= upper_band or rsi_value > 70:
                self.log("Shorting due to overbought conditions.")
                self.order_target_value(target=0)

            # Fallback: Ensure position is 0 just before expiration time
            if current_date == third_thursday and self.data.datetime.time(0) >= datetime.time(14, 30):
                self.log("Fallback: Forcing position to 0 at the end of expiration week.")
                self.order_target_value(target=0)


    def prepare_features(self):
        """Prepare features for PCA and KMeans training."""
        if len(self.data.close) < self.params.train_window:
            return None

        closes = np.array(self.data.close.get(size=self.params.train_window))
        returns = np.diff(np.log(closes))  # Log returns

        atr_values = np.array([self.atr[-i] for i in range(self.params.train_window)])
        rsi_values = np.array([self.rsi[-i] for i in range(self.params.train_window)])
        ma_fast_values = np.array([self.ma_fast[-i] for i in range(self.params.train_window)])
        ma_slow_values = np.array([self.ma_slow[-i] for i in range(self.params.train_window)])

        if len(returns) == 0 or len(atr_values) == 0:
            self.log("Error: Empty data during feature preparation. Skipping...")
            return None

        features = np.column_stack([returns, atr_values[:-1], rsi_values[:-1], ma_fast_values[:-1], ma_slow_values[:-1]])
        return features

    def next(self):
        # Log current data
        self.log(
            f"Close: {self.data.close[0]:.2f}, "
            f"Portfolio Value: {self.broker.getvalue():.2f}"
        )

        # Get the current time
        current_date = self.data.datetime.datetime(0)
        current_hour, current_min = current_date.hour, current_date.minute
        current_time = datetime.time(hour=current_hour, minute=current_min)

        # Define restricted periods
        mor_start, mor_end = datetime.time(8, 45), datetime.time(9, 0)
        noon_start, noon_end = datetime.time(11, 0), datetime.time(13, 0)
        aft_cutoff = datetime.time(14, 0)

        # Check if current time is within restricted periods
        if (
            (mor_start <= current_time < mor_end) or
            (noon_start <= current_time < noon_end) or
            (current_time >= aft_cutoff)
        ):
            self.log(f"Trading restricted at: {current_time}")
            return

        # Determine if it's expiration week and short if conditions are met
        current_date = self.data.datetime.date(0)
        third_thursday = self.get_third_thursday(current_date)
        expiration_week_start = third_thursday - datetime.timedelta(days=4)

        if expiration_week_start <= current_date <= third_thursday:
            self.find_best_short_timing()
            return

        # Prepare features
        features = self.prepare_features()
        if features is None:
            self.log("Insufficient data for training. Skipping...")
            return

        # Train PCA and KMeans
        try:
            pca_features = self.pca.fit_transform(features)
            self.kmeans.fit(pca_features)
            current_cluster = self.kmeans.predict(pca_features[-1].reshape(1, -1))[0]
        except Exception as e:
            self.log(f"Training Error: {e}")
            return

        self.log(f"Cluster: {current_cluster}")

        # Trade logic
        portfolio_value = self.broker.getvalue()

        if current_cluster == 0:  # Bullish cluster
            self.order_target_value(target=-portfolio_value * self.params.trade_fraction)
        elif current_cluster == 1:  # Bearish cluster
            self.order_target_value(target=portfolio_value * self.params.trade_fraction)
        elif current_cluster == 2:  # Neutral cluster
            self.order_target_value(target=portfolio_value * self.params.trade_fraction)


def calculate_date_range():
    today = datetime.datetime.now()
    start_of_this_month = today.replace(day=1)
    todate = start_of_this_month - relativedelta(months=0)
    fromdate = todate - relativedelta(months=3)
    return fromdate, todate


if __name__ == '__main__':
    fromdate, todate = calculate_date_range()
    print(f"From Date: {fromdate}")
    print(f"To Date: {todate}")

    simulation = rbt.RopyLab(plot=True)

    simulation.add_multi_api_data(
        symbols=['VN30F1M'],
        timeframe='15m',
        fromdate=fromdate,
        todate=todate,
        key = "a8c15e63376357f2aa70722b2f33f67f2a9d613ce2107e8f70e4c5c9286f34d1"
    )

    simulation.set_cash(1e9)
    simulation.addstrategy(AdaptivePCAKMeansAlpha)
    simulation.run(output='alpha_test.csv', value_output=True, trade_output=False)
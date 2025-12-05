from __future__ import (absolute_import, division, print_function, unicode_literals)

import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import rework_backtrader as rbt
from rework_backtrader.indicators import (RSI, AverageTrueRange, SimpleMovingAverage)


class AdaptivePCAKMeansAlpha(rbt.rbtstrategies.RopylabStrategy):
    params = dict(
        atr_period=14,
        rsi_period=14,
        ma_fast_period=10,
        ma_slow_period=30,
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

        # Models
        self.pca = PCA(n_components=self.params.pca_components)
        self.kmeans = KMeans(n_clusters=self.params.k_clusters, random_state=42)

    def prepare_features(self):
        """
        Prepare features for PCA and KMeans training.
        """
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

        # Prepare features
        features = self.prepare_features()
        if features is None:
            self.log("Insufficient data for training. Skipping...")
            return
        
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
        position_size = self.getposition().size

        if current_cluster == 0:  # Bullish cluster
            if self.data.close[0] > self.data.close[-1]:
                self.log("Bullish cluster detected. Going long.")
                self.order_target_value(target=-portfolio_value * self.params.trade_fraction)
        elif current_cluster == 1:  # Bearish cluster
            if self.data.close[0] < self.data.close[-1]:
                self.log("Bearish cluster detected. Going short.")
                self.order_target_value(target=portfolio_value * self.params.trade_fraction)
        elif current_cluster == 2:  # Neutral cluster
            self.log("Neutral cluster detected. Closing positions.")
            self.order_target_value(target=portfolio_value * self.params.trade_fraction)



def calculate_date_range():
    today = datetime.datetime.now()
    start_of_this_month = today.replace(day=1)
    todate = start_of_this_month - relativedelta(months=3)
    fromdate = todate - relativedelta(months=18)
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
        todate=todate
    )

    simulation.set_cash(1e9)
    simulation.addstrategy(AdaptivePCAKMeansAlpha)
    simulation.run(output='alpha_test.csv', value_output=True, trade_output=False)

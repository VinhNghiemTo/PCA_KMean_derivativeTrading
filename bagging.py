from __future__ import (absolute_import, division, print_function, unicode_literals)

import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import rework_backtrader as rbt
from rework_backtrader.indicators import RSI, AverageTrueRange, SimpleMovingAverage, BollingerBands


class EnsembleAlpha(rbt.rbtstrategies.RopylabStrategy):
    params = dict(
        lookback_period=30,         # Longer lookback for trend detection
        training_window=100,        # Training samples
        atr_period=14,              # ATR period for risk management
        rsi_period=14,              # RSI period
        sma_period=50,              # SMA for trend filtering (new)
        trade_fraction=0.5,         # Lower trade fraction (was 1.0)
        n_estimators=200,           # More trees for stability
        prediction_threshold=0.05,  # Higher threshold (was 0.01) to avoid noise
        atr_multiplier=2.0,         # ATR-based stop loss & take profit
        bb_period=20
    )

    def __init__(self):
        super(EnsembleAlpha, self).__init__()
        self.order = None

        # Indicators
        self.atr = AverageTrueRange(self.data, period=self.params.atr_period)
        self.rsi = RSI(self.data.close, period=self.params.rsi_period)
        self.sma = SimpleMovingAverage(self.data.close, period=self.params.sma_period)  # TREND FILTER

        self.bb = BollingerBands(self.data.close, period=self.params.bb_period)

        # Model and scaler
        self.model = RandomForestRegressor(
            n_estimators=self.params.n_estimators,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.trained = False

    def prepare_features(self):
        """Prepare features for training and prediction."""
        available_data = len(self.data.close)
        lookback = min(self.params.lookback_period, available_data - 1)

        if lookback < 5:
            self.log(f"Insufficient data for preparation. Available: {available_data}, Required: {self.params.lookback_period}")
            return None, None

        closes = np.array(self.data.close.get(size=lookback + 1))
        returns = np.diff(np.log(closes))  

        atr_values = np.array([self.atr[-i] for i in range(1, lookback + 1)])
        rsi_values = np.array([self.rsi[-i] for i in range(1, lookback + 1)])
        sma_trend = np.array([self.sma[-i] for i in range(1, lookback + 1)])

        # Add momentum feature
        momentum = returns - np.mean(returns)

        # Align lengths
        min_length = min(len(returns), len(atr_values), len(rsi_values), len(momentum), len(sma_trend))
        returns, atr_values, rsi_values, momentum, sma_trend = [arr[:min_length] for arr in [returns, atr_values, rsi_values, momentum, sma_trend]]

        # Normalize and return
        X = np.column_stack([returns, atr_values, rsi_values, momentum, sma_trend])
        y = np.sign(returns)  
        X = self.scaler.fit_transform(X)  

        return X, y

    def train_model(self):
        """Train the model with the latest data."""
        X, y = self.prepare_features()
        if X is None:
            self.log("Insufficient data for training.")
            return

        training_window = min(self.params.training_window, len(X))

        if training_window < 5:
            self.log(f"Insufficient training data. Available: {len(X)}, Required: {self.params.training_window}")
            return

        self.model.fit(X[-training_window:], y[-training_window:])
        self.trained = True
        self.log(f"Model trained successfully on {training_window} samples.")

        # Log feature importance
        feature_importance = self.model.feature_importances_
        self.log(f"Feature Importances: {feature_importance}")

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

    def next(self):
        """Perform trading logic."""
        self.log(f"Date: {self.data.datetime.date(0)}, Portfolio Value: {self.broker.getvalue():.2f}")

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

        # Train the model if it's not trained yet
        if not self.trained:
            self.train_model()

        if not self.trained:
            self.log("Model not trained yet. Skipping.")
            return

        # Prepare the latest features for prediction
        X, _ = self.prepare_features()
        if X is None:
            self.log("Insufficient data for prediction. Skipping.")
            return

        latest_features = X[-1].reshape(1, -1)
        prediction = self.model.predict(latest_features)[0]
        self.log(f"Prediction: {prediction:.5f}")

        portfolio_value = self.broker.getvalue()
        atr_value = self.atr[0] * self.params.atr_multiplier

        # Apply TREND FILTER - Only trade in trend direction
        if self.data.close[0] > self.sma[0]:  # Uptrend
            if prediction > self.params.prediction_threshold:
                self.log("Trend Up. Going Long.")
                self.order_target_value(target=portfolio_value * self.params.trade_fraction)
            elif prediction < -self.params.prediction_threshold:
                self.log("Trend Up but predicted down. No trade.")
        else:  # Downtrend
            if prediction < -self.params.prediction_threshold:
                self.log("Trend Down. Going Short.")
                self.order_target_value(target=-portfolio_value * self.params.trade_fraction)
            elif prediction > self.params.prediction_threshold:
                self.log("Trend Down but predicted up. No trade.")

        # Add stop-loss and take-profit
        self.log(f"Setting Stop-Loss at {self.data.close[0] - atr_value:.2f}, Take-Profit at {self.data.close[0] + atr_value:.2f}")


def calculate_date_range():
    today = datetime.datetime.now()
    start_of_this_month = today.replace(day=1)
    todate = start_of_this_month - relativedelta(months=3)
    fromdate = todate - relativedelta(months=3)
    return fromdate, todate


if __name__ == '__main__':
    fromdate, todate = calculate_date_range()
    print(f"From Date: {fromdate}")
    print(f"To Date: {todate}")

    simulation = rbt.RopyLab(plot=True)

    simulation.add_multi_api_data(
        symbols=["VN30F1M"],
        timeframe='30m',
        fromdate=fromdate,
        todate=todate
    )

    simulation.set_cash(1e9)
    simulation.addstrategy(EnsembleAlpha)
    simulation.run(output='alpha_test.csv', value_output=True, trade_output=False)

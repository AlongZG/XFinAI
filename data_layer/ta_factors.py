import talib as tl


class TaFactor:
    def __init__(self, data):
        self.data = data.copy(deep=True)
        self.open = self.data['open'].values
        self.close = self.data['close'].values
        self.high = self.data['high'].values
        self.low = self.data['low'].values
        self.money = self.data['money'].values
        self.volume = self.data['volume'].values
        self.open_interest = self.data['open_interest'].values

    def calc_bbands(self):
        for tick in [5, 10, 30, 90, 180, 270, 365]:
            for threshold in [1.5, 2, 2.5]:
                self.data[f'BBANDS_{tick}tick_{threshold}STD_upper'], self.data[
                    f'BBANDS_{tick}tick_{threshold}STD_middle'], \
                self.data[f'BBANDS_{tick}tick_{threshold}STD_lower'] = tl.BBANDS(self.close, nbdevup=threshold,
                                                                                 nbdevdn=threshold, timeperiod=tick)

    def calc_ma(self):
        for tick in [5, 10, 30, 90, 180, 270, 365]:
            self.data[f'DEMA_{tick}tick'] = tl.DEMA(self.close, timeperiod=tick)
            self.data[f'EMA_{tick}tick'] = tl.EMA(self.close, timeperiod=tick)
            self.data[f'SMA_{tick}tick'] = tl.SMA(self.close, timeperiod=tick)
            self.data[f'KAMA_{tick}tick'] = tl.KAMA(self.close, timeperiod=tick)
            self.data[f'WMA_{tick}tick'] = tl.WMA(self.close, timeperiod=tick)
            self.data[f'TRIMA_{tick}tick'] = tl.TRIMA(self.close, timeperiod=tick)

    def calc_momentum(self):
        # Momentum类
        for tick in [5, 10, 30, 90, 180, 270, 365]:
            self.data[f'ADX_{tick}tick'] = tl.ADX(self.high, self.low, self.close,
                                                  timeperiod=tick)
            self.data[f'AROONOSC_{tick}tick'] = tl.AROONOSC(self.high, self.low, timeperiod=tick)
            self.data[f'DX_{tick}tick'] = tl.DX(self.high, self.low, self.close,
                                                timeperiod=tick)
            self.data[f'MFI_{tick}tick'] = tl.MFI(self.high, self.low, self.close,
                                                  self.volume,
                                                  timeperiod=tick)
            self.data[f'MOM_{tick}tick'] = tl.MOM(self.close, timeperiod=tick)
            self.data[f'RSI_{tick}tick'] = tl.RSI(self.close, timeperiod=tick)
            self.data[f'WILLR_{tick}tick'] = tl.WILLR(self.high, self.low, self.close,
                                                      timeperiod=tick)

    def calc_mass(self):
        self.data[f'MACD_macd'], self.data[f'MACD_macdsignal'], self.data[f'MACD_macdhist'] = tl.MACD(self.close)
        self.data[f'MAMA_mama'], self.data[f'MAMA_fama'] = tl.MAMA(self.close)
        self.data[f'HT_TRENDLINE'] = tl.HT_TRENDLINE(self.close)
        self.data[f'SAR'] = tl.SAR(self.high, self.low)
        self.data[f'SAREXT'] = tl.SAR(self.high, self.low)
        self.data[f'ULTOSC'] = tl.ULTOSC(self.high, self.low, self.close)
        self.data[f'BOP'] = tl.BOP(self.open, self.high, self.low, self.close)
        self.data[f'APO'] = tl.APO(self.close)
        self.data[f'PPO'] = tl.PPO(self.close)

    def run(self):
        self.calc_bbands()
        self.calc_ma()
        self.calc_momentum()
        self.calc_mass()
        return self.data

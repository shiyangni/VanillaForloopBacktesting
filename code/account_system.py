# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:54:27 2020

@author: Shiyang
"""

#%% 

import numpy as np
import pandas as pd
import os
import warnings
warnings.simplefilter('ignore')
os.chdir(os.path.dirname(__file__))
from enum import Enum
from OHLC_arbitrary_frequency import add_time

#%%
class Signal(Enum):
    BUY = 1
    HOLD = 0
    SELL = -1

class Account:
    ### An account reads in a trade history, and reflects the positions(size and values) in assets
    ### and liabilities in a clear fashion.
    ### trade_history needs to contain at least time, price, signal, asset_bought, asset_sold,
    ### and amount_bought, amount_sold, 
    def __init__(self, transaction_history, initial_equity, inital_equity_exists_as='cash'):
        time = transaction_history['time'] 
        self.transaction_history = transaction_history
        self.assets = pd.DataFrame({'time': time})
        self.liabilities = pd.DataFrame({'time': time})
        self.equities = pd.DataFrame({'time': time})
        self.equities['initial_equity'] = initial_equity
        self.assets[inital_equity_exists_as] = initial_equity
    
    def process_transaction_history(self):
        for i in np.arange(self.transaction_history.shape[0]):
            this_transaction = self.transaction_history.iloc[i, :]
            if this_transaction['signal'] == Signal.HOLD:
                continue
            else:
                this_asset_bought = this_transaction['asset_bought']
                this_asset_sold = this_transaction['asset_sold']
                this_amount_bought = this_transaction['amount_bought']
                this_amount_sold = this_transaction['amount_sold']
                this_time = this_transaction['time']
                self._add_amount_to_asset(this_asset_bought, this_amount_bought, this_time)
                self._add_amount_to_asset(this_asset_sold, -this_amount_sold, this_time)
        return None
                
    def show_assets_with_transactions(self, target_asset, base_asset):
        result = self.assets.merge(self.transaction_history[['time', 'price', 'signal', 'position_suggested']], how='inner', on='time')
        result['position_in_base'] = result[target_asset] * result['price'] + result[base_asset] - self.transaction_history['commission'].values.cumsum()
        return result        
    
    def _add_amount_to_asset(self, asset_to_add_to, amount, time):
        ### This utility function adds the specified amount to the designated asset account.
        ### If the asset account already exists, we add to the existing position, and set
        ### the position on the remaining days to the resulting value.
        ### If the asset account doesn't exist, we add create a new column in self.assets with
        ### the designated asset as its name, and a starting position of zero.
        if asset_to_add_to in self.assets.columns:
            self.assets.loc[self.assets['time'] >= time, asset_to_add_to] += amount
        else:
            self.assets[asset_to_add_to] = 0
            self.assets.loc[self.assets['time'] >= time, asset_to_add_to] += amount
        return None


class SignalGenerator:
    def __init__(self, price_df):
        self.price_df = price_df
    
    def get_price_with_signal(self):
        self.price_df['signal'] = Signal.HOLD
        return self.price_df
        
class NaiveSignalGenerator(SignalGenerator):
    
    def get_price_with_signal(self):
        self.price_df['signal'] = np.random.choice(Signal, self.price_df.shape[0])
        return self.price_df
        
class PositionSizer:
    def __init__(self, df_price_signal):
        self.df = df_price_signal
        
    def get_price_signal_position(self):
        return self.df
    
class NaivePositionSizer(PositionSizer):
    def __init__(self, df_price_signal, position_size=100):
        self.df = df_price_signal
        self.position_size = position_size
    
    def get_price_signal_position(self):
        self.df['position_suggested'] = 0
        self.df.loc[self.df['signal'].apply(lambda x: x.value) == Signal.BUY.value,'position_suggested'] = self.position_size
        sell_part = self.df.loc[self.df['signal'].apply(lambda x: x.value) == Signal.SELL.value, :]
        for i in sell_part.index:
            sell_amount = -self.df.loc[self.df.index < i, 'position_suggested'].sum()
            self.df.loc[i, 'position_suggested'] = sell_amount
        self.df['position_suggested'] = self.df['position_suggested'].abs()
        return self.df
    

class ExecutionHandler:
    def __init__(self, df_price_signal_position, target_asset, base_asset):
        self.df = df_price_signal_position
        self.target_asset = target_asset
        self.base_asset = base_asset
        
    def get_execution_details(self, price_actual, commission_rate):
        self._add_asset_bought_sold()
        self._add_price_actual(price_actual)
        self._add_amount_bought_and_sold()
        self._add_commission(commission_rate)
        return self.df
    
        
    def _add_asset_bought_sold(self):
        self.df['asset_bought'] = self.target_asset
        self.df['asset_bought'][self.df['signal'].apply(lambda x: x.value) == Signal.SELL.value] = self.base_asset
        self.df['asset_sold'] = self.base_asset
        self.df['asset_sold'][self.df['signal'].apply(lambda x: x.value) == Signal.SELL.value] = self.target_asset
        return self.df
    
    def _add_price_actual(self, price_actual):
        self.df['price_actual'] = price_actual
        return self.df
    
    def _add_amount_bought_and_sold(self):
        ### Assuming the function is always executed after the price_actual is added.
        self.df['amount_bought'] = 0
        self.df['amount_sold'] = 0
        sell_part_index = self.df['signal'].apply(lambda x: x.value) == Signal.SELL.value
        buy_part_index = self.df['signal'].apply(lambda x: x.value) == Signal.BUY.value
        self.df.loc[buy_part_index, 'amount_bought'] = self.df.loc[buy_part_index, 'position_suggested']
        self.df.loc[buy_part_index, 'amount_sold'] = self.df.loc[buy_part_index, 'amount_bought'] * self.df.loc[buy_part_index, 'price_actual']
        self.df.loc[sell_part_index, 'amount_sold'] = self.df.loc[sell_part_index, 'position_suggested']
        self.df.loc[sell_part_index, 'amount_bought'] = self.df.loc[sell_part_index, 'amount_sold'] * self.df.loc[sell_part_index, 'price_actual']
        return self.df
    
    def _add_commission(self, commission_rate=0.025):
        self.df['commission'] = 0
        sell_part_index = self.df['signal'].apply(lambda x: x.value) == Signal.SELL.value
        buy_part_index = self.df['signal'].apply(lambda x: x.value) == Signal.BUY.value
        self.df.loc[sell_part_index, 'commission'] = self.df.loc[sell_part_index, 'amount_sold']  * commission_rate
        self.df.loc[buy_part_index, 'commission'] = self.df.loc[buy_part_index, 'amount_bought']  * commission_rate
        return self.df

def extract_price_data(df_OHLC, price_name='close'):
    df_OHLC.pipe(add_time)
    df_price = df_OHLC[['time', price_name]].copy()
    df_price.rename(columns = {price_name: 'price'}, inplace=True)
    return df_price
    
    
def naively_handle(df_price, signal_generator_constructor, 
                   naive_position_sizer_constructor, naive_execution_handler_constuctor,
                   position_size, target_asset, base_asset, price_actual, 
                   commission_rate=0.025):
    signal_generator = signal_generator_constructor(df_price)
    df_signal = signal_generator.get_price_with_signal()
    position_sizer = naive_position_sizer_constructor(df_signal, position_size=position_size)
    df_position = position_sizer.get_price_signal_position()
    naive_execution_handler = naive_execution_handler_constuctor(df_position, target_asset, base_asset)
    df_execution = naive_execution_handler.get_execution_details(price_actual=price_actual, commission_rate=commission_rate)
    return df_execution


# #%%
# ### Preparing data
# BCH_BTC = pd.read_csv('bch-btc_OHLC.csv')
# price_naive = BCH_BTC['close']



# BCH_BTC_with_execution = naively_handle(BCH_BTC.pipe(extract_price_data), NaiveSignalGenerator,
#                                         NaivePositionSizer, ExecutionHandler, position_size=100,
#                                         target_asset='BCH', base_asset='BTC', price_actual=price_naive
#                                         , commission_rate=0.025)



# #%%


# BCH_BTC_with_execution_mini = BCH_BTC_with_execution.head(5000)
# test_account = Account(BCH_BTC_with_execution_mini, 0, 'BTC')

# tick = time.time()
# test_account.process_transaction_history()
# result = test_account.show_assets_with_transactions()
# tock = time.time()
# elapsed = tock - tick
# print('Time elapsed during transaction processing: {}s'.format(round(elapsed, 2)))
    





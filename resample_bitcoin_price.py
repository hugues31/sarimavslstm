from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
import numpy as np
import pandas as pd


def create_dataframe():
    frame = pd.read_csv(askopenfilename(), header=None)
    frame.columns = ['Date', 'Prix', 'Montant']
    frame['Date'] = pd.to_datetime(frame['Date'], unit='s')
    frame.set_index(frame['Date'], inplace=True)
    frame = frame[['Prix', 'Montant']]
    return frame


def get_bitcoin_ohlc(frame, freq='1D'):
    """
    Compute bitcoin OHLC data frame with the given frequency.
    @param frame: raw bitcoin price history.
    @param freq: target OHLC frequency.
    """
    mean = frame['Prix'].resample(freq).mean().ffill()
    # for column in ['open', 'high', 'low', 'close']:
    #     ohlc[column] = np.where(np.isnan(ohlc[column]), close, ohlc[column])
    # mean['amount'] = frame['amount'].resample(freq, how='last')
    # mean['amount'].fillna(0.0, inplace=True)
    return mean


df = create_dataframe()
ohlc = get_bitcoin_ohlc(df, freq='1H')

# on enregistre les changements en pourcentage
ohlc.pct_change().to_csv(asksaveasfilename(
    defaultextension='.csv'), index=None, header=True)

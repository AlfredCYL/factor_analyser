import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib import cm
from typing import Literal

class evaluation_manager():
    def __init__(
        self,
        factor:pd.DataFrame| pd.Series, 
        returns:pd.DataFrame| pd.Series,
        forward_time:int = 0 ) -> pd.DataFrame | pd.Series:

        # 只保留共有的列
        factor = factor.loc[factor.index.intersection(returns.index),factor.columns.intersection(returns.columns)]
        returns = returns.reindex_like(factor)

        self._factor = factor.copy()
        self._f_mean = factor.mean(axis=1)
        self._f_std = factor.std(axis=1)
        self._factor_normalized = self._factor.apply(lambda x: (x - self._f_mean[x.name]) /self._f_std[x.name] , axis=1)
        
        self._returns = returns.copy().shift(-forward_time)

        self._method = None
        self._weights = None
        self._profit = None
        self._ic = None
        self._quantile_returns = None

    @staticmethod
    def performance_summary(
        profit,
        annualization=252
    ):
        return_data = pd.DataFrame(profit,columns=["factor_return"])

        summary_stats = return_data.mean().to_frame('Mean').apply(lambda x: x*annualization)
        summary_stats['Volatility'] = return_data.std().apply(lambda x: x*np.sqrt(annualization))
        summary_stats['Sharpe Ratio'] = summary_stats['Mean']/summary_stats['Volatility']
        
        summary_stats['Skewness'] = return_data.skew()
        summary_stats['Excess Kurtosis'] = return_data.kurtosis()
        summary_stats['VaR (0.05)'] = return_data.quantile(.05, axis = 0)
        summary_stats['CVaR (0.05)'] = return_data[return_data <= return_data.quantile(.05, axis = 0)].mean()
        summary_stats['Min'] = return_data.min()
        summary_stats['Max'] = return_data.max()
        
        wealth_index = 1000*(1+return_data).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks)/previous_peaks

        summary_stats['Max Drawdown'] = drawdowns.min()
        summary_stats['Peak'] = [previous_peaks[col][:drawdowns[col].idxmin()].idxmax() for col in previous_peaks.columns]
        summary_stats['Bottom'] = drawdowns.idxmin()
        
        recovery_date = []
        for col in wealth_index.columns:
            prev_max = previous_peaks[col][:drawdowns[col].idxmin()].max()
            recovery_wealth = pd.DataFrame([wealth_index[col][drawdowns[col].idxmin():]]).T
            recovery_date.append(recovery_wealth[recovery_wealth[col] >= prev_max].index.min())
        summary_stats['Recovery'] = recovery_date
        
        return summary_stats

    @staticmethod
    def get_factor_signs(
        factors,
    ):
        return  np.sign(factors)
    
    @staticmethod
    def neutralize_weights(
        weights: pd.DataFrame,
        ignore:int = 0
    ):
        res = weights.copy()

        if ignore == 0:
            pass
        elif ignore == -1:
            res = res.mask(res>0,other=0)
        elif ignore == 1:
            res = res.mask(res<0,other=0)      

        return res.div(res.abs().sum(axis = 1), axis='index')
    
    def _calculate_factor_weights(
        self,
        method:Literal['long_short_equal_weights','short_equal_weights','long_equal_weights','long_short','long','short']  = 'long_short',
        ):
        if method == 'long_equal_weights':
            self._weights = evaluation_manager.neutralize_weights(evaluation_manager.get_factor_signs(self._factor_normalized))

        elif method == 'short_equal_weights':
            self._weights = evaluation_manager.neutralize_weights(evaluation_manager.get_factor_signs(self._factor_normalized),ignore=-1)
            
        elif method == 'long_short_equal_weights':
            self._weights = evaluation_manager.neutralize_weights(evaluation_manager.get_factor_signs(self._factor_normalized),ignore=1)
            
        elif method == 'long_short':
            self._weights = evaluation_manager.neutralize_weights(self._factor_normalized)
            
        elif method == 'short':
            self._weights = evaluation_manager.neutralize_weights(self._factor_normalized,ignore=-1)
            
        elif method == 'long':
            self._weights = evaluation_manager.neutralize_weights(self._factor_normalized,ignore=1)
        
        self._method = method

    def _calculate_factor_returns(
        self,
        ):
        self._profit = self._weights.mul(self._returns).sum(axis = 1)


    def _calculate_ic(self,rankIC:bool = True):
        if rankIC:
            self._ic = self._factor.corrwith(self._returns,axis = 1, method = "spearman")
        else:
            self._ic = self._factor.corrwith(self._returns,axis = 1, method = "pearson")
    
    
    def _calculate_quantile_returns(self, quantiles: int = 10):
        groups = np.array(range(quantiles)) + 1

        factor_quantiles = self._factor.rank(axis=1,method='first').apply(pd.qcut, q=quantiles, labels = groups,axis=1)
        
        # Step 2: Iterate over unique groups
        return_series = {}

        for group in groups:
            returns_group = self._returns[factor_quantiles == group]
            return_series[group] = returns_group.sum(axis=1) / returns_group.count(axis=1) # scale holding to 1 ; equal weights

        self._quantile_returns = return_series


    def get_ic(
        self,
        plot:bool = False
    ):
        if self._ic is None:
            self._calculate_ic()
        if plot:
            self.plot_ic()

        return self._ic
    

    def get_factor_rets(
        self,
        method:Literal['long_short_equal_weights','short_equal_weights','long_equal_weights','long_short','long','short'] = "long_short",  
        plot:bool = False
    ):
        if method != self._method:
            self._calculate_factor_weights(method)
            self._calculate_factor_returns()

        if plot:
            self.plot_factor_rets()
        return self._profit
        
    def get_quantile_rets(
        self,
        quantiles:int = 10,
        plot:bool = False
    ):
        if self._quantile_returns is None or len(self._quantile_returns) != quantiles:
            self._calculate_quantile_returns(quantiles)

        if plot:
            self.plot_quantile_returns()
        return self._quantile_returns
    
    def plot_ic(
        self,
        freq:str = 'W',
        rolling_win: int = 4
    ):
        if self._ic is None:
            self._calculate_ic()
        ic = self._ic.copy()
        ic.index = pd.to_datetime(ic.index)

        # 将时间序列按周进行降频，并对每周的值进行平均处理
        s_resample_mean = ic.resample(freq).mean()
        s_resample_mean_swm = s_resample_mean.rolling(rolling_win).mean()

        # 创建一个figure实例，指定图形大小
        fig = plt.figure(figsize=(15,5))
        # 在第一行上绘制散点图和滑动平均线
        colors = np.where(abs(s_resample_mean.values) > 0.2, 'darkred', 
            np.where(abs(s_resample_mean.values) > 0.1, 'red', 
            np.where(abs(s_resample_mean.values) > 0.03, 'blue', 'gray')))
        plt.scatter(s_resample_mean.index, s_resample_mean.values, s=50, c=colors)
        plt.plot(s_resample_mean_swm.index, s_resample_mean_swm.values, linewidth=3, linestyle='-', color='gray', alpha=0.8)
        plt.axhline(0, linestyle='--', color='gray', alpha=0.5)
        plt.title(freq + ' Frequent Mean of ICs')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.xlim(s_resample_mean.index.min(), s_resample_mean.index.max())

        # 创建一个figure实例，指定图形大小
        fig, axs = plt.subplots(ncols=3, figsize=(15,4))
        # 在第二个axes上绘制直方图
        axs[0].hist(ic.values, color='cornflowerblue', alpha=0.5)
        axs[0].set_title('Distribution of IC')
        axs[0].set_xlabel('Values')
        axs[0].set_ylabel('Frequency')

        # 在第二个axes上绘制QQ图
        sm.qqplot(ic.values, line='s', ax=axs[1])
        axs[1].set_title('QQ Plot of IC')
        axs[1].set_xlabel('Theoretical Quantiles')
        axs[1].set_ylabel('Sample Quantiles')

        # 在第三个axes上绘制cumulative ICs
        axs[2].plot(ic.cumsum())
        axs[2].set_title('Cumulative ICs')
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Values')
        # 调整x轴刻度
        axs[2].xaxis.set_major_locator(mdates.AutoDateLocator())
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # 显示图形
        plt.show()
        
    def plot_factor_rets(
        self,
        method:Literal['long_short_equal_weights','short_equal_weights','long_equal_weights','long_short','long','short'] = "long_short",  
    ):
        if method != self._method:
            self._calculate_factor_weights(method)
            self._calculate_factor_returns()
        
        pnl = self._profit.copy()
        pnl.index = pd.to_datetime(pnl.index)
        pnl = pd.DataFrame(pnl, columns=[self._method])

        fig, ax1 = plt.subplots(figsize=(15,5))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.plot(pnl.loc[:, self._method].cumsum(), label='Price', c='blue')
        ax1.set_title(f"Factor Returns: {self._method}")
        ax1.legend()
        plt.show()

    def plot_quantile_returns(
        self,
        quantiles:int = 10,
        annualization = 252
    ):

        if self._quantile_returns is None or len(self._quantile_returns) != quantiles:
            self._calculate_quantile_returns(quantiles)

        pnl = self._quantile_returns.copy()

        pnl = pd.DataFrame(pnl)
        pnl.index = pd.to_datetime(pnl.index)

        my_colors = cm.seismic(np.arange(quantiles) / quantiles )

        # 创建一个figure实例，指定图形大小
        fig, axs = plt.subplots(ncols=2, figsize=(15,4))

        # 在第一个axes上绘制收益曲线，使用渐变色
        pnl.cumsum().plot(ax=axs[0], color=my_colors, alpha=0.5)
        axs[0].set_title('Quantile Returns of IC')
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Cumulative Returns')

        # 在第二个axes上绘制bar图，使用渐变色
        pnl.mean().to_frame('Mean').apply(lambda x: x*annualization).plot.bar(ax=axs[1], color='blue')
        axs[1].set_title('Quantile Returns of IC')
        axs[1].set_xlabel('Quantiles')
        axs[1].set_ylabel('Annualized Returns')


    def evaluate_pnl(self,annualization=252):
        return evaluation_manager.performance_summary(self._profit,annualization)

    
    
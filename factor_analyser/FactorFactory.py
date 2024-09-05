import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import plotly.graph_objects as go
import networkx as nx
from IPython.display import display, HTML
import statsmodels.api as sm
from scipy.stats import gaussian_kde
import os

from .FactorBacktester import FactorBacktester

class FactorFactory:
    def __init__(self, returns, forward_time = 0):
        returns = returns.copy().shift(-forward_time).iloc[:-forward_time] if forward_time > 0 else returns.copy() # forward_ret
        returns.index = pd.to_datetime(returns.index)
        self.stock_ret = returns  

    def evaluate_factor(self, factor, num_quantiles=20, factor_ret_method='long_short'):
        """Evaluates a factor's comprehensive performance."""
        null_percentage = self.calculate_null_percentage(factor)
        factor.dropna(axis=0, thresh=1, inplace=True) # drop rows with all NaNs
        
        bt = FactorBacktester(factor, self.stock_ret)
        ic = bt.get_ic(plot=True)
        mean_ic = ic.mean()
        std_ic = ic.std()
        icir = mean_ic / std_ic
        print(f'rank_ic: {mean_ic:.4f}, rank_ic_std: {std_ic:.4f}, rank_ic_ir: {icir:.4f}, null_percentage: {null_percentage}')

        quantile_rets = bt.get_quantile_rets(quantiles=num_quantiles, plot=True, evaluation=True)
        factor_ret = bt.get_factor_rets(method=factor_ret_method, plot=True, evaluation=True)

    def evaluate_factor_icir(self, factor, icir_thresh=0.3, ic_thresh=0.025, return_only = False):
        """Evaluates a factor based on IC and ICIR thresholds with formatting."""
        null_percentage = self.calculate_null_percentage(factor)
        factor.dropna(axis=0, thresh=1, inplace=True) # drop rows with all NaNs

        bt = FactorBacktester(factor, self.stock_ret)
        ic = bt.get_ic(plot=False)
        mean_ic = ic.mean()
        std_ic = ic.std()
        icir = mean_ic / std_ic

        if not return_only:
            content = f'rank_ic: {mean_ic:.4f}, rank_ic_std: {std_ic:.4f}, rank_ic_ir: {icir:.4f}, null_percentage: {null_percentage}'
            if abs(icir) > icir_thresh and abs(mean_ic) > ic_thresh:
                display(HTML(f'<p style="color: darkred;">{content}</p>'))
            elif abs(icir) > icir_thresh or abs(mean_ic) > ic_thresh:
                display(HTML(f'<p style="color: salmon;">{content}</p>'))
            else:
                display(HTML(f'<p style="color: gray;">{content}</p>'))
        
        return icir, mean_ic, std_ic, null_percentage

    def evaluate_factor_extension_stats(self, factor, window = 20, min_periods = 2):
        """quickly evaluates various extensions of a factor."""
        print(f"{window}Mean:")
        self.evaluate_factor_icir(factor.rolling(window,min_periods=min_periods).mean())
        print(f"{window}RankMean:")
        self.evaluate_factor_icir(factor.rank(1).rolling(window,min_periods=min_periods).mean())
        print(f"{window}ExpMean:")
        self.evaluate_factor_icir(factor.rolling(window,min_periods=min_periods,win_type = 'exponential').mean(center=window, tau = 1/np.log(1 - (2 / (window))), sym=False))
        print(f"{window}Std:")
        self.evaluate_factor_icir(factor.rolling(window,min_periods=min_periods).std())
        print(f"{window}Skew:")
        self.evaluate_factor_icir(factor.rolling(window,min_periods=min_periods).skew())
        print(f"{window}Kurt:")
        self.evaluate_factor_icir(factor.rolling(window,min_periods=min_periods).kurt())
        print(f"{window}Median:")
        self.evaluate_factor_icir(factor.rolling(window,min_periods=min_periods).median())
        print(f"{window}Q25:")
        self.evaluate_factor_icir(factor.rolling(window,min_periods=min_periods).quantile(0.25))
        print(f"{window}Q75:")
        self.evaluate_factor_icir(factor.rolling(window,min_periods=min_periods).quantile(0.75))

    def save_factor(self, factor, factor_name="factor", time_point = "1100", root_path = "./data/factors/"):
        """Saves the factor data to a specified directory."""
        factor_dir = os.path.join(root_path, factor_name,time_point)
        os.makedirs(factor_dir, exist_ok=True)
        factor_save_path = os.path.join(factor_dir, "data.parq")
        factor.index.name = "date"
        factor.columns.name = "symbol"
        factor_df = factor.stack().to_frame(factor_name).reset_index()  # Unstack and reset index
        factor_df.to_parquet(factor_save_path)

    def plot_heatmap(self, df, plot_type = 'heatmap' ,title='Heatmap', cmap='coolwarm', annot=True, fmt=".2f", figsize=(10, 8), vmin=-1, vmax=1):
        """Plots a heatmap for the given DataFrame."""
        if plot_type == 'heatmap':
            plt.figure(figsize=figsize)
            sns.heatmap(df, annot=annot, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.title(title)
        elif plot_type == 'clustermap':
            g = sns.clustermap(df, annot=annot, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax)
            g.figure.suptitle(title)
        plt.show()

    def find_correlated_groups(self, df, threshold=0.6):
        """Finds groups of correlated features in the given correlation table."""
        G = nx.Graph()
        G.add_nodes_from(df.columns)
        
        for i in range(len(df.columns)):
            for j in range(i + 1, len(df.columns)):
                if df.iloc[i, j] > threshold:
                    G.add_edge(df.columns[i], df.columns[j])
        
        subgraphs = []
        for component in nx.connected_components(G):
            if len(component) > 1:
                subgraphs.append(G.subgraph(component))
            else:
                subgraphs.append(component)
        
        groups = {}
        for i, subgraph in enumerate(subgraphs):
            if isinstance(subgraph, set):
                groups[f"Group {i+1}"] = list(subgraph)
            else:
                groups[f"Group {i+1}"] = list(subgraph.nodes())
        
        return groups
    
    def factor_correlation_analysis(self, factors, factor_names=None, return_all_matrices=False, plot=True, corr_method='spearman', sample_size = None, plot_type='heatmap', annot = True, fmt=".2f", cmap='coolwarm', figsize=(10, 8)):
        """Analyzes correlation between multiple factors."""
        if factor_names is None:
            factor_names = [f"factor_{i}" for i in range(len(factors))]
        factor_df = pd.concat([f.stack().rename(factor_names[i]) for i, f in enumerate(factors)], axis=1)
        factor_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if sample_size is not None:
            if sample_size > factor_df.shape[0]:
                raise ValueError("Sample size should be less than the number of observations.")
            factor_df = factor_df.sample(sample_size)

        correlation_matrices = factor_df.groupby(level=0).corr(method=corr_method)
        mean_correlation = correlation_matrices.groupby(level=1).mean().loc[factor_names, factor_names]
        if plot:
            self.plot_heatmap(mean_correlation, plot_type= plot_type, title='Factor Correlation', cmap=cmap, annot=annot, fmt=fmt, figsize=figsize)
        return (correlation_matrices, mean_correlation) if return_all_matrices else mean_correlation

    def calculate_factor_overlap(self, df1, df2, low_pct=0.8, high_pct=1):
        """Analyzes stock selection overlap between two factors based on df1."""
        assert df1.shape == df2.shape, "DataFrames should have the same dimensions."
    
        low_quantile_1 = df1.quantile(low_pct, axis=1)
        high_quantile_1 = df1.quantile(high_pct, axis=1)
        low_quantile_2 = df2.quantile(low_pct, axis=1)
        high_quantile_2 = df2.quantile(high_pct, axis=1)

        mask_1 = df1.apply(lambda x: (x >= low_quantile_1[x.name]) & (x <= high_quantile_1[x.name]), axis=1).astype(int)
        mask_2 = df2.apply(lambda x: (x >= low_quantile_2[x.name]) & (x <= high_quantile_2[x.name]), axis=1).astype(int)
        
        overlap_mask = mask_1 & mask_2
        overlap_scores = overlap_mask.sum(axis=1) / mask_1.sum(axis=1)
        return overlap_scores

    def plot_joint_pdf_with_outlier_removal(self, series1, series2, name1 = 'f1', name2 = 'f2', lower_bound = 0, upper_bound = 1, interactive = False):
        df = pd.concat([series1, series2], axis=1)
        df.columns = [name1, name2]
        lower_bound_x = df[name1].quantile(lower_bound)
        upper_bound_x = df[name1].quantile(upper_bound)
        lower_bound_y = df[name2].quantile(lower_bound)
        upper_bound_y = df[name2].quantile(upper_bound)

        filtered_df = df[(df[name1] >= lower_bound_x) & (df[name1] <= upper_bound_x) & (df[name2] >= lower_bound_y) & (df[name2] <= upper_bound_y)]

        x = filtered_df[name1].values
        y = filtered_df[name2].values

        kde = gaussian_kde(np.vstack([x, y]))

        x_fine = np.linspace(x.min(), x.max(), 100)
        y_fine = np.linspace(y.min(), y.max(), 100)
        xgrid_fine, ygrid_fine = np.meshgrid(x_fine, y_fine)

        z_values_fine = kde(np.vstack([xgrid_fine.ravel(), ygrid_fine.ravel()]))
        zgrid_fine = np.reshape(z_values_fine, xgrid_fine.shape)
        if interactive:
            fig = go.Figure(data=[go.Surface(z=zgrid_fine, x=xgrid_fine, y=ygrid_fine)])
            fig.update_layout(title='Joint Probability Density', autosize=True,
                            scene=dict(
                                xaxis_title=f'{name1}',
                                yaxis_title=f'{name2}',
                                zaxis_title='Probability Density'),
                            margin=dict(l=65, r=50, b=65, t=90))
            fig.show()
        else:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            ax.plot_surface(xgrid_fine, ygrid_fine, zgrid_fine, cmap='viridis')

            ax.set_xlabel(f'{name1}')
            ax.set_ylabel(f'{name2}')
            ax.set_zlabel('Probability Density')
            plt.show()

    def plot_distribution(self, series, bins='auto', log_scale=False):
        """Plots the distribution of a series with optional log scaling."""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(series, bins=bins, kde=False, log_scale=log_scale, ax=ax)
        if log_scale:
            ax.set_xscale('log')
        ax.grid(True)
        ax.legend(labels=['Frequency'])
        ax.set_title('Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        plt.show()
 
    def plot_factor_quantiles(self, factor, use_plotly=False, log_scale=False, figsize=(12, 8)):
        """Plots the quantiles of a factor from a dynamic perspective."""
        quantiles = pd.concat([factor.quantile(i / 100, axis=1) for i in range(0, 105, 10)], axis=1)
        quantiles.columns = [f'{i}%' for i in range(0, 105, 10)]
        
        if use_plotly:
            fig = go.Figure()
            for col in quantiles.columns:
                fig.add_trace(go.Scatter(x=quantiles.index, y=quantiles[col], mode='lines', name=col))
            fig.update_layout(
                title="Quantiles of Test Factor",
                xaxis_title="Date",
                yaxis_title="Value",
                yaxis_type='log' if log_scale else 'linear',
                margin=dict(l=20, r=20, t=20, b=20)
            )
            fig.show()
        else:
            colors = cm.viridis(np.linspace(0, 1, len(quantiles.columns)))
            fig, ax = plt.subplots(figsize=figsize)
            for col, color in zip(quantiles.columns, colors):
                ax.plot(quantiles.index, quantiles[col], label=col, color=color)
            ax.set_title("Quantiles of Test Factor")
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            if log_scale:
                ax.set_yscale('log')
            csm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=100))
            csm.set_array([])
            cbar = fig.colorbar(csm, ax=ax, pad=0.01, ticks=np.linspace(0, 100, len(quantiles.columns)))
            cbar.set_label("Percentiles")
            plt.tight_layout()  # 为了更好地适应Legend和颜色条
            plt.show()

    def calculate_null_percentage(self, df):
        """Calculates the average percentage of NaNs in the DataFrame."""
        null_percentage = df.isnull().mean() * 100
        return null_percentage.mean()
    
    def standardize_factor(self, factor):
        """Standardizes the factor data by z-scoring across each day."""
        mean = factor.mean(axis=1)
        std = factor.std(axis=1)
        standardized_factor = factor.sub(mean, axis=0).div(std, axis=0)
        return standardized_factor
    
    def replace_outliers(self, df, num_std=3, winsorize = False):
        """Replaces outliers in DataFrame with NaN or n-sigma boundaries, based on a specified number of standard deviations."""
        df = df.copy() # Avoid modifying the original DataFrame
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        
        if winsorize:
            row_means = mean.to_numpy()
            row_stds = std.to_numpy()
            
            upper_limits = row_means[:, np.newaxis] + (num_std * row_stds)[:, np.newaxis]
            lower_limits = row_means[:, np.newaxis] - (num_std * row_stds)[:, np.newaxis]
            
            df_capped = np.clip(df, lower_limits, upper_limits, axis=1)
            return pd.DataFrame(df_capped, columns=df.columns, index=df.index)
        else:
            is_outlier = np.abs(df.sub(mean, axis=0)) > (num_std * std).values.reshape(-1, 1)
            df[is_outlier] = np.nan
            return df

    def compute_daily_residuals_simple(self, y, x):
        """Computes residuals from a linear regression between two DataFrames."""
        assert y.shape == x.shape, "DataFrames should have the same dimensions."
        residuals_df = pd.DataFrame(index=y.index, columns=y.columns)
        for index in y.index:
            X = x.loc[index].values
            Y = y.loc[index].values
            X = sm.add_constant(X)
            valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(Y)
            X_valid = X[valid_indices]
            Y_valid = Y[valid_indices]
            if len(Y_valid) > 1:
                model = sm.OLS(Y_valid, X_valid).fit()
                residuals = model.resid
                residuals_df.loc[index, valid_indices] = residuals
            else:
                residuals_df.loc[index] = np.nan
        return residuals_df

    def linear_regression_residuals(self, y, X, categorical_columns, drop_first=True, dummy_na=False):
        """Computes residuals from linear regressions between provided DataFrames."""
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=drop_first, dummy_na=dummy_na)
        X = sm.add_constant(X)
        model = sm.OLS(y, X.astype('float')).fit()
        return model.resid
    
    def compute_daily_residuals(self, df_y, continuous_Xs, continuous_X_names=[], categorical_Xs=[], categorical_X_names=[], drop_first=True):
        """Computes daily cross-sectional residuals from linear regressions between provided DataFrames."""
        residuals_df = pd.DataFrame(index=df_y.index, columns=df_y.columns)

        if len(continuous_X_names) != len(continuous_Xs):
            continuous_X_names = [f'f_{i}' for i in range(1, 1 + len(continuous_Xs))]
        if len(categorical_Xs) > 0 and len(categorical_X_names) != len(categorical_Xs):
            categorical_X_names = [f'cat_{i}' for i in range(1, 1 + len(categorical_Xs))]
        
        for index in df_y.index:
            Y = df_y.loc[index]
            X_sub = pd.concat([df_x.loc[index] for df_x in continuous_Xs], axis=1)
            X_sub.columns = continuous_X_names
            
            if len(categorical_Xs) > 0:
                X_catg = pd.concat([df_x.loc[index] for df_x in categorical_Xs], axis=1)
                X_catg.columns = categorical_X_names
                X_sub = pd.concat([X_sub, X_catg], axis=1)

            X_sub = sm.add_constant(X_sub)
            valid_indices = ~np.isnan(X_sub[continuous_X_names]).any(axis=1) & ~np.isnan(Y)  # 只检查连续变量和Y
            X_valid = X_sub.loc[valid_indices]
            Y_valid = Y[valid_indices]
            dummy_na = X_sub[categorical_X_names].isnull().any().any()
            if len(Y_valid) > 1:
                residuals = self.linear_regression_residuals(Y_valid, X_valid, categorical_X_names, dummy_na=dummy_na, drop_first=drop_first)
                residuals_df.loc[index, valid_indices] = residuals
            else:
                residuals_df.loc[index] = np.nan
        return residuals_df
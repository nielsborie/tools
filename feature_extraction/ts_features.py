def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Taken from: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Forecast storage
forecasts = {k: [] for k in ['ElasticNet_idx', 'RandomForest_lags3']}

# Run various forecasts
for i, row in tqdm(total_df[ordered_cols].iterrows(), total=len(total_df)):
    
    # Linear regression 2 steps into the future based on index
    regr = ElasticNetCV(cv=LeaveOneOut())
    regr.fit(
        np.arange(0, len(row)).reshape(-1, 1), 
        row.values.reshape(-1, 1).ravel()
    )
    forecasts['ElasticNet_idx'].append(regr.predict([[len(row)+2]])[0])
    
    # Random forest based on 3 lagged features
    sdf = series_to_supervised(row.values.tolist(), 3, 2)
    regr = ExtraTreesRegressor(n_estimators=10)
    regr.fit(sdf[['var1(t-3)', 'var1(t-2)', 'var1(t-1)']], sdf['var1(t+1)'])
    forecasts['RandomForest_lags3'].append(
        regr.predict([row.values[-3:]])[0]
    )
    
# put into dataframe
forecasts_df = pd.DataFrame(forecasts)
aggregate_df = pd.DataFrame()

# Wrapper function
def diff2(x):
    return np.diff(x, n=2)

# Different pre-processing to be used before each primary function
preprocess_steps = [
    [],
    [np.diff], [diff2],
    [np.unique], [np.unique, np.diff], [np.unique, diff2]    
]

# Different statistics to calculate on each preprocessed step
stats = [len, np.min, np.max, np.median, np.std, skew, kurtosis] + 19 * [np.percentile]
stats_kwargs = [{} for i in range(7)] + [{'q': np.round(i, 2)} for i in np.linspace(0.05, 0.95, 19)]

# Only operate on non-nulls
for funs in preprocess_steps:
    
    # Apply pre-processing steps
    x = total_df[total_df != 0]
    for f in funs:
        x = f(x)
        
    # Go through our set of stat functions
    for stat, stat_kwargs in zip(stats, stats_kwargs):
        
        # Construct feature name
        name_components = [
            stat.__name__,
            "_".join([f.__name__ for f in funs]),
            "_".join(["{}={}".format(k, v) for k,v in stat_kwargs.items()])
        ]
        feature_name = "-".join([e for e in name_components if e])

        # Calc and save new feature in our dataframe
        aggregate_df[feature_name] = total_df.apply(lambda x: stat(x, **stat_kwargs), axis=1)
        
# Extra features
n0_df = total_df[total_df>0]

# Features on all columns
aggregate_df['n0_count'] = total_df.astype(bool).sum(axis=1)
aggregate_df['n0_mean'] = n0_df.mean(axis=1)
aggregate_df['n0_median'] = n0_df.median(axis=1)
aggregate_df['n0_kurt'] = n0_df.kurt(axis=1)
aggregate_df['n0_min'] = n0_df.min(axis=1)
aggregate_df['n0_std'] = n0_df.std(axis=1)
aggregate_df['n0_skew'] = n0_df.skew(axis=1)
aggregate_df['mean'] = total_df.mean(axis=1)
aggregate_df['median'] = total_df.median(axis=1)
aggregate_df['std'] = total_df.std(axis=1)
aggregate_df['max'] = total_df.max(axis=1)
aggregate_df['min'] = total_df.min(axis=1)
aggregate_df['nunique'] = total_df.nunique(axis=1)
aggregate_df['number_of_different'] = total_df.nunique(axis=1)
aggregate_df['non_zero_count'] = total_df.astype(bool).sum(axis=1) 
aggregate_df['sum_zeros'] = (total_df == 0).astype(int).sum(axis=1)
aggregate_df['non_zero_fraction'] = total_df.shape[1] / total_df.astype(bool).sum(axis=1) 
aggregate_df['geometric_mean'] = total_df.apply(
    lambda x: np.exp(np.log(x[x>0]).mean()), axis=1
)
aggregate_df.reset_index(drop=True, inplace=True)
aggregate_df['geometric_mean'] = aggregate_df['geometric_mean'].replace(np.nan, 0)
aggregate_df['non_zero_fraction'] = aggregate_df['non_zero_fraction'].replace(np.inf, 0)

# Show user which aggregates were created
print(f">> Created {len(aggregate_df.columns)} features for; {aggregate_df.columns.tolist()}")
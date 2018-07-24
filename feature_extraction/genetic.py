from gplearn.genetic import SymbolicTransformer

function_set = ['add', 'sub', 'mul', 'div',
                'inv', 'log', 'abs', 'neg', 
                'sqrt', 'max', 'min']

gp = SymbolicTransformer(
    generations=10, population_size=50000,
    hall_of_fame=100, n_components=10,
    function_set=function_set,
    parsimony_coefficient=0.0005,
    max_samples=0.9, verbose=1,
    random_state=42, n_jobs=4
)

# Fit & save to dataframe
gp.fit(total_df.iloc[train_idx], y)
gp_features = gp.transform(total_df)
genetic_df = pd.DataFrame(gp_features, columns=[f'Genetic_{i}' for i in range(gp_features.shape[1])])
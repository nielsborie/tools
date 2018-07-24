COMPONENTS = 10

# Convert to sparse matrix
sparse_matrix = scipy.sparse.csr_matrix(X.values)

# Data to be passed to t-SNE
tsvd = TruncatedSVD().fit_transform(sparse_matrix)

# List of decomposition methods
methods = [
    {'method': NMF(n_components=COMPONENTS,random_state=5), 'data': 'total'},
    {'method': FactorAnalysis(n_components=COMPONENTS,random_state=5), 'data': 'log'},
    {'method': multi_TSNE(n_components=3, init='random',n_jobs=15), 'data': 'total'},
    {'method': TruncatedSVD(n_components=COMPONENTS,random_state=5), 'data': 'sparse'},
    {'method': PCA(n_components=COMPONENTS,random_state=5), 'data': 'total'},
    {'method': FastICA(n_components=COMPONENTS,random_state=5), 'data': 'total'},
    {'method': GaussianRandomProjection(n_components=COMPONENTS, eps=0.1,random_state=5), 'data': 'total'},
    {'method': SparseRandomProjection(n_components=COMPONENTS, dense_output=True,random_state=5), 'data': 'total'}
]

# Run all the methods
embeddings = []
for run in methods:
    name = run['method'].__class__.__name__
    
    # Run method on appropriate data
    if run['data'] == 'sparse':
        embedding = run['method'].fit_transform(sparse_matrix)
    elif run['data'] == 'tsvd':
        embedding = run['method'].fit_transform(tsvd)
    elif run['data'] == 'log':
        embedding = run['method'].fit_transform(np.log1p(X))
    else:
        embedding = run['method'].fit_transform(X)
        
    # Save in list of all embeddings
    embeddings.append(
        pd.DataFrame(embedding, columns=[str(name)+"_"+str(i) for i in range(embedding.shape[1])])
    )
    print(">> Ran {}".format(name))
    gc.collect()    
    
# Put all components into one dataframe
components_df = pd.concat(embeddings, axis=1).reset_index(drop=True)
print()
print("Created {} decomposition features\n".format(components_df.shape[1]))
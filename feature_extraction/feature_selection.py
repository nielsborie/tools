# Train / test bool target
y_clf = np.zeros(len(total_df))
y_clf[test_idx] = 1

def get_safe_KS(df, thr=0.1):
    """Use KS to determine columns with KS statistic above threshold between train & test"""

    # Book-keeping
    drop_features = []

    # Go through all columns
    with tqdm() as pbar:
        for col in feature_df.columns:

            # Columns left
            cols_left = [c for c in feature_df.columns if c not in drop_features]
            pbar.update(1)

            # Look at distribution in feature
            statistic, pvalue = ks_2samp(
                feature_df.loc[train_idx, col].values, 
                feature_df.loc[test_idx, col].values
            )
            if pvalue < 0.05 and statistic > 0.1:
                pbar.set_description(f"Dropping: {col}. KS: {statistic}. p-value: {pvalue}. {len(cols_left)} features left.")
                drop_features.append(col)
            
    # Return columns to keep
    return cols_left

def get_safe_adversarial(df, thr=0.7):
    """Recursively eliminate features from adversarial validation with highest feature importance,
    Continues untill the accuracy of the oob-score for a random forest decreases below given threshold
    """
    
    # Book-keeping
    current_score = np.inf
    drop_features = []
    
    # Start eliminating features
    with tqdm() as pbar:
        while current_score > thr:

            # Columns left
            cols_left = [c for c in df.columns if c not in drop_features]

            # Fit random forest model
            regr = ExtraTreesClassifier(n_estimators=100, oob_score=True, bootstrap=True)
            regr.fit(df[cols_left], y_clf)
            current_score = regr.oob_score_
            pbar.update(1)

            # Get most important feature for classification
            best_feature = cols_left[np.argmax(regr.feature_importances_)]

            # Add to drop and inform user
            if current_score > thr:
                pbar.set_description(f"Acc: {regr.oob_score_}. Dropping: {best_feature}.")
                drop_features.append(best_feature)
            else:
                pbar.set_description(f"Adversarial Elimination reached threshold acc of {thr}. {len(cols_left)} features left.")
    return cols_left

# Create plot for KS elimination
cols_left = get_safe_KS(feature_df, 0.1)
plot_feature_scores(results, feature_list=cols_left, title="After Kolmogorov–Smirnov feature elimination")

# Create plot for adversarial elimination
cols_left = get_safe_adversarial(feature_df, 0.7)
plot_feature_scores(results, feature_list=cols_left, title="After adversarial feature elimination")

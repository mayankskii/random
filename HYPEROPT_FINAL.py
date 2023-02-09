#%pip install lightgbm hyperopt shap lime optbinning scikit-plot pandas-profiling mlflow fastparquet openpyxl imblearn pyod pip -U -q
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pandasql import sqldf

from optbinning import OptimalBinning, ContinuousOptimalBinning
from tqdm import tqdm_notebook as tqdm


# Data Filter
c1 = (data2['claim_repudiation_ind']==0) | (data2['claim_repudiation_ind'].isna())
c2 = (data2['buildings_claim_ind']==1) | (data2['buildings_claim_ind'].isna())
c3 = (data2['buildings_incurred_amt']>0) | (data2['buildings_incurred_amt'].isna())
c4 = data2['epol_total']>(1/12)
c5 = data2['written_exposure']>0
c6 = ~data2['policy_number'].duplicated()

c_name = ['claim_repudiation_ind==0', 'buildings_claim_ind==1', 'buildings_incurred_amt>0', 'epol_total>1/12', 'written_exposure>0', 'policy_duplicates (propensity only)']

# Generating filter wise Reduction
dfs=[]
data_f = data2.copy()
data_f['peril_desc'] = data_f['peril_desc'].cat.add_categories(['no_claim']).fillna('no_claim')
aggfunc = {'policy_number': 'nunique', 'claim_number': 'nunique'}
df = pd.pivot_table(data_f, values=aggfunc.keys(), \
           index=['peril_desc'], aggfunc=aggfunc, fill_value=0, margins=True, dropna=False, margins_name='Total').reset_index()
df = df.set_index('peril_desc').T
df.loc['event_rate', :] = df.loc['policy_number', :]/df.loc['policy_number', 'Total']
df.loc['event_rate', 'Total'] = df.loc['claim_number', 'Total']/df.loc['policy_number', 'Total']
df = df.reset_index().rename(columns={'index': 'col_stat'})
df.insert(loc=0, column='filter', value='no_filter')
dfs.append(df)
for i, j in zip([c1, c2, c3, c4, c5, c6], c_name):
    data_f = data_f.loc[i].reset_index(drop=True)
    aggfunc = {'policy_number': 'nunique', 'claim_number': 'nunique'}
    df = pd.pivot_table(data_f, values=aggfunc.keys(), \
               index=['peril_desc'], aggfunc=aggfunc, fill_value=0, margins=True, dropna=False, margins_name='Total').reset_index()
    df = df.set_index('peril_desc').T
    df.loc['event_rate', :] = df.loc['policy_number', :]/df.loc['policy_number', 'Total']
    df.loc['event_rate', 'Total'] = df.loc['claim_number', 'Total']/df.loc['policy_number', 'Total']
    df = df.reset_index().rename(columns={'index': 'col_stat'})
    df.insert(loc=0, column='filter', value=j)
    dfs.append(df)
dfs = pd.concat(dfs)

data3 = data2.loc[c1&c2&c3&c4&c5, [col for col in data2.columns if col in idx+house_attrs+['ph_age_as_of_cutoff_date', 'property_purchase_age', 'property_occupied_age', 'postcode_area', 'claim']+\
                            ['Fire', 'Flood', 'Impact - Fallen Trees', 'Storm', 'Subsidence']]]



# Binning Features
def get_binning_metrics(binning_table, x, y):
    bin_stats = binning_table.build()
    binning_table.plot(metric='event_rate', savefig=f'/home/ec2-user/SageMaker/mayank/models/propensity/plots/{y.name}/event_rate/{x.name}.png')
    binning_table.plot(metric='woe', savefig=f'/home/ec2-user/SageMaker/mayank/models/propensity/plots/{y.name}/woe/{x.name}.png')
    binning_table.analysis(print_output=False)
    bin_stats = bin_stats.reset_index().rename(columns={'index': 'Bin ID'})
    bin_stats.insert(loc=0, column='Feature', value=x.name)

    name = binning_table.name
    count = x.shape[0]
    n_event = y.sum()
    event_rate = y.mean()
    p_missing = x.isna().mean()
    gini = binning_table.gini
    iv = binning_table.iv
    js = binning_table.js
    hell = binning_table.hellinger
    tri = binning_table.triangular
    ks = binning_table.ks
    hhi = binning_table._hhi
    hhi_norm = binning_table._hhi_norm
    quality_score = binning_table.quality_score
    return bin_stats, pd.DataFrame(pd.Series({'Feature': name, 'Count': count, 'Event': n_event, 'Event rate': event_rate, 'Missing %': p_missing,
                             'Gini index' : gini, 'IV (Jeffrey)' : iv, 'JS (Jensen-Shannon)' : js, 'Hellinger' : hell,
                             'Triangular' : tri, 'KS' : ks, 'HHI' : hhi, 'HHI (normalised)' : hhi_norm, 'Quality score' : quality_score})).T
def get_optimal_binning(data, variable, target, optb):
    x, y = data[variable], data3[target]
    optb.fit(x, y)
    binning_table = optb.binning_table
    bin_stats, agg_bin_stats = get_binning_metrics(binning_table, x, y)
    return optb, binning_table, bin_stats, agg_bin_stats

# Binning Features Execution
target = 'Storm'
optb_fitted = {'optb': {}, 'binning_table': {}, 'bin_stats': {}, 'agg_bin_stats': {}}

numerical_no_ops = ['ph_age', 'alarm_types', 'buildings_ncd', 'contents_ncd', 'listed_building', 'unoccupied_days', 'year_of_construction', 'flat_roof_percentage',
                    'number_of_occupants', 'number_of_floors', 'buildings_area', 'contents_area', 'buildings_ad_area', 'buildings_eow_area', 'buildings_flood_area',
                    'buildings_storm_area', 'buildings_subsidence_area', 'buildings_theft_area', 'buildings_fire_area', 'buildings_other_area', 'buildings_market_area',
                    'contents_ad_area', 'contents_eow_area', 'contents_flood_area', 'contents_storm_area', 'contents_theft_area', 'contents_fire_area', 
                    'contents_other_area', 'contents_market_area', 'contents_region_area', 'price_test_area', 'dwelling_type', 'ownership_type', 'est_floor_area', 'number_of_days',
                    'type_of_alarm', 'heating_method_primary', 'number_of_bedrooms', 'total_number_of_rooms', 'ph_age_as_of_cutoff_date', 'property_purchase_age', 'property_occupied_age']
for i in tqdm(numerical_no_ops):
    optb = OptimalBinning(name=i, dtype="numerical", special_codes=None)
    optb, binning_table, bin_stats, agg_bin_stats = get_optimal_binning(data3, i, target, optb)
    optb_fitted['optb'][i] = optb
    optb_fitted['binning_table'][i] = binning_table
    optb_fitted['bin_stats'][i] = bin_stats
    optb_fitted['agg_bin_stats'][i] = agg_bin_stats
    
num_99_ops = ['number_of_adults', 'number_of_bathrooms', 'roofcode', 'wallcode', 'number_of_children', 'eligibility_indicator', 'buildings_tenure',
              'contents_tenure', 'number_rooms_used_for_business', 'pets_at_home_flag']
for i in tqdm(num_99_ops):
    optb = OptimalBinning(name=i, dtype="numerical", special_codes=[99])
    optb, binning_table, bin_stats, agg_bin_stats = get_optimal_binning(data3, i, target, optb)
    optb_fitted['optb'][i] = optb
    optb_fitted['binning_table'][i] = binning_table
    optb_fitted['bin_stats'][i] = bin_stats
    optb_fitted['agg_bin_stats'][i] = agg_bin_stats
              
cat_99_ops = ['employerbusiness', 'marital_status', 'occupation_code', 'ownership', 'property_type', 'swh', 'occupancy_status', 'relationship_to_main_ph',
              'ph_occupation', 'connells_res_let_flag', 'employment_status', 'postcode_area', 'conviction_flag', 'subject_to_an_iva_flag', 'fire_precautions_code',
              'security_device_desc', 'good_repair_flag', 'stock_on_premises_flag', 'security_area_code', 'landslip_damage_flag', 'ground_heave_damage_flag',
              'building_underpinned']
for i in tqdm(cat_99_ops):
    optb = OptimalBinning(name=i, dtype="categorical", special_codes=[99])
    optb, binning_table, bin_stats, agg_bin_stats = get_optimal_binning(data3, i, target, optb)
    optb_fitted['optb'][i] = optb
    optb_fitted['binning_table'][i] = binning_table
    optb_fitted['bin_stats'][i] = bin_stats
    optb_fitted['agg_bin_stats'][i] = agg_bin_stats
    
    
y_n_99_ops = ['clerical_business_use', 'first_time_buyer', 'free_from_underpinning', 'freesubs', 'neighbourhood_watch_scheme', 'property_extended',
              'smoke_alarms', 'smoker', 'unoccupied_at_night', 'door_locks_flag', 'window_locks_flag', 'bankruptcy_flag', 'used_for_business_flag',
              'client_visits_premises_flag', 'near_riverbank_quarry_cliff_flag', 'simplification_migration_flag',
              'instalment_flag', 'prev_poms_hh_polholder_flag', 'prev_poms_travel_polholder_flag', 'approved_alarm_fitted', 'alarm_spec_y_n', 'lockable_entrance',
              'standard_construction', 'occupied_by_family', 'residential', 'clerical_business_flag', 'occupancy_over_30_days', 'signed_prop_in_possess',
              'maint_agree_in_force', 'safe_installed', 'retire_accom_with_warden']
for i in tqdm(y_n_99_ops):
    optb = OptimalBinning(name=i, dtype="categorical", special_codes=[99], user_splits=[['N'], ['Y'], ['U'], ['X']])
    optb, binning_table, bin_stats, agg_bin_stats = get_optimal_binning(data3, i, target, optb)
    optb_fitted['optb'][i] = optb
    optb_fitted['binning_table'][i] = binning_table
    optb_fitted['bin_stats'][i] = bin_stats
    optb_fitted['agg_bin_stats'][i] = agg_bin_stats

drop_cols = ['convictioncode', 'ownership_years', 'selfcontained', 'number_of_other_rooms', 'post_office_score_1', 'post_office_score_2', 'post_office_score_3', 
             'post_office_score_4', 'post_office_score_5', 'post_office_score_6', 'post_office_score_7', 'post_office_score_8', 'post_office_score_9', 'post_office_score_10',
             'contents_value', 'high_risk_sum_insured', 'estimated_property_value', 'frozen_food_flag', 'payment_method', 'mobile_phone_section', 'money_and_credit_cards',
             'caravan_type_section', 'subsidence_damage', 'heating_method_2']

# Binning Summary
binning_summary = pd.concat(optb_fitted['agg_bin_stats'], ignore_index=True)
binning_detailed = pd.concat(optb_fitted['bin_stats'], ignore_index=True)
binning_summary.to_csv(f'/home/ec2-user/SageMaker/mayank/models/propensity/optbin/{target}_binning_summary.csv', index=False)
binning_detailed.to_csv(f'/home/ec2-user/SageMaker/mayank/models/propensity/optbin/{target}_binning_detailed.csv', index=False)


# Propensity Model
targets = ['Fire', 'Flood', 'Impact - Fallen Trees', 'Storm', 'Subsidence']
target = 'Storm'

data4 = data5[idx+numerical_no_ops+num_99_ops+cat_99_ops+y_n_99_ops+[target]].reset_index(drop=True)
data4['postcode_area'] = data4['postcode_area'].astype('category')


# Model Objective
import numpy as np
import pandas as pd
import pickle

import lightgbm as lgbm

from functools import partial
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK

from sklearn.model_selection import cross_validate, train_test_split, learning_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, DetCurveDisplay

import scikitplot as skplt
import matplotlib.pyplot as plt

import shap

def format_float(a):
    if isinstance(a, float) and (a).is_integer():
            return int(a)
    else: return a
    
def objective(space, clf, presets, X_train, y_train, cv=3, scoring='average_precision', delta=0.05, std=0.0002):
    params = {k:format_float(v) for k, v in {**space,**presets}.items()}
    CV = cross_validate(clf.set_params(**params), X_train, y_train, cv=cv, scoring=scoring, error_score='raise', return_train_score=True)
    
    train_score = np.mean(CV['train_score'])
    test_score = np.mean(CV['test_score'])
    x = np.abs(train_score - test_score)
    
    f_val = np.piecewise(x, [x < delta, x >= delta], [test_score, test_score*(np.exp(-0.5*((x-delta)**2)/std))])
    return {'loss': -f_val, 'status': STATUS_OK}

presets = {
    'boosting_type': 'dart',
    'objective': 'binary',
    'metric': 'average_precision',
    'random_state': 42,
    'class_weight': 'balanced' 
}

space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.05), np.log(0.25)),
    'num_leaves': hp.quniform('num_leaves', 8, 128, 2),
    'n_estimators': hp.quniform('n_estimators', 100, 300, 2),
    'max_depth': hp.quniform('max_depth', 3, 12, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0)
}
assert ~bool(set(presets.keys()) & set(space.keys()))

# Train Test Splits
X, y = data4.loc[:, ~data4.columns.isin([target])], data4.loc[:, target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

segregate_cols = idx+[target]
X_train = X_train[[col for col in X_train.columns if col not in segregate_cols]]
X_test = X_test[[col for col in X_test.columns if col not in segregate_cols]]


# Training Model
fmin_objective = partial(objective, clf=lgbm.LGBMClassifier(), presets=presets, X_train=X_train, y_train=y_train)
trials = Trials()
arg_min = fmin(fn=fmin_objective, algo=tpe.suggest, space=space, max_evals=32, trials=trials)

# Calibrated Model
local_minima = {k:format_float(v) for k, v in {**arg_min, **presets}.items()}
model = lgbm.LGBMClassifier().set_params(**local_minima)
calibrated_clf = CalibratedClassifierCV(base_estimator=model, cv=3)
model.fit(X_train, y_train)
calibrated_clf.fit(X_train, y_train)

# Save Pickle
with open(f"/home/ec2-user/SageMaker/mayank/models/propensity/models_pkl/{target}/sklearn-lgbm-propensity.pkl", "wb") as f:
    pickle.dump(model, f)
with open(f"/home/ec2-user/SageMaker/mayank/models/propensity/models_pkl/{target}/sklearn-lgbm_claibrated-propensity.pkl", "wb") as f:
    pickle.dump(calibrated_clf, f)
    
    
# Model Summary
model_summary = lgbm.LGBMClassifier().set_params(**local_minima)
model_summary.fit(X_train, y_train, eval_set=[(X_test, y_test), (X_train, y_train)], eval_metric=['mape', 'poisson', 'auc', 'average_precision', \
                                                                                                  'binary_logloss', 'binary_error', 'cross_entropy', 'kullback_leibler'], verbose=-1)

for i in ['mape', 'poisson', 'auc', 'average_precision', 'binary_logloss', 'binary_error', 'cross_entropy', 'kullback_leibler']:
    lgbm.plot_metric(model_summary, metric=i, figsize=(16,9)).get_figure().savefig(f"/home/ec2-user/SageMaker/mayank/models/propensity/model_summary/{target}/learning_curves/{i}.png")

lgbm.plot_importance(model, figsize = (16,9), max_num_features=25).get_figure().savefig(f"/home/ec2-user/SageMaker/mayank/models/propensity/model_summary/{target}/statistics/importance.png")

y_pred = model.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_pred, figsize = (16,9)).get_figure().savefig(f"/home/ec2-user/SageMaker/mayank/models/propensity/model_summary/{target}/statistics/roc.png")
skplt.metrics.plot_ks_statistic(y_test, y_pred, figsize = (16,9)).get_figure().savefig(f"/home/ec2-user/SageMaker/mayank/models/propensity/model_summary/{target}/statistics/ks_statistic.png")
skplt.metrics.plot_precision_recall(y_test, y_pred, figsize = (16,9)).get_figure().savefig(f"/home/ec2-user/SageMaker/mayank/models/propensity/model_summary/{target}/statistics/precision_recall.png")
skplt.metrics.plot_cumulative_gain(y_test, y_pred, figsize = (16,9)).get_figure().savefig(f"/home/ec2-user/SageMaker/mayank/models/propensity/model_summary/{target}/statistics/cummulative_gain.png")
skplt.metrics.plot_lift_curve(y_test, y_pred, figsize = (16,9)).get_figure().savefig(f"/home/ec2-user/SageMaker/mayank/models/propensity/model_summary/{target}/statistics/lift_curve.png")

det_curve = DetCurveDisplay.from_predictions(y_test, y_pred[:, 1], name=f'{target}_propensity').figure_
det_curve.set_size_inches((16,9))
det_curve.savefig(f"/home/ec2-user/SageMaker/mayank/models/propensity/model_summary/{target}/statistics/det_curve.png")

pd.DataFrame(classification_report(y_test, y_pred[:, 1]>0.5, output_dict=True)).to_csv(f"/home/ec2-user/SageMaker/mayank/models/propensity/model_summary/{target}/statistics/classification_report.csv")



# Explaining the Model
shap_values = shap.TreeExplainer(model).shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, plot_type='dot', plot_size=(16,9), show=False)
plt.savefig(f"/home/ec2-user/SageMaker/mayank/models/propensity/model_summary/{target}/shap/summary_plot.png")

X_test_cat = X_test.copy()
cols = X_test_cat.select_dtypes('category').columns
X_test_cat[cols] = X_test_cat[cols].apply(lambda x: x.cat.codes)

for i in pd.Series(model.feature_name_, index=model.feature_importances_).sort_index(ascending=False)[:20].values:
    shap.dependence_plot(i, shap_values[1], X_test_cat.values, feature_names=X_test.columns)
    
    
# Experiment with a variable
variable = 'landslip_damage_flag'
pd.concat([data3[variable].value_counts(dropna=False, normalize=True), data3[variable].value_counts(dropna=False, normalize=False)], axis=1)
x, y = data3[variable], data3.claim

# optb = OptimalBinning(name=variable, dtype="numerical", special_codes=None)
# optb = OptimalBinning(name=variable, dtype="numerical", special_codes=[98, 99])
optb = OptimalBinning(name=variable, dtype="categorical", special_codes=[99])
# optb = OptimalBinning(name=variable, dtype="categorical", special_codes=[99], user_splits=[['N'], ['Y'], ['U'], ['X']])
optb.fit(x, y)

binning_table = optb.binning_table
binning_table.build()

binning_table.plot(metric='woe')
binning_table.analysis()

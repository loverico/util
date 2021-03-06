import pandas as pd
import math
from boruta import BorutaPy
def balance_positive_rate(rdf, columns=[], positive_ratio=None, random_state=None):
    df = bq_gasessions_default(rdf, columns=columns)

    print('positive_ratio: {}, random_state: {}'.format(positive_ratio, random_state))

    if positive_ratio is None:
        raise ValueError('positive_ratio must be set.')

    df_plabel = df[df.label == POSITIVE_LABEL]
    df_nlabel = df[df.label == NEGATIVE_LABEL]
    num_plabel = df_plabel.index.size
    num_nlabel = df_nlabel.index.size
    if not num_plabel + num_nlabel == df.index.size:
        raise ValueError('unexpected state.')
    num_sampled_nlabel = min(num_nlabel, math.floor(num_plabel / positive_ratio))
    sampled_nlabel = df_nlabel.sample(n=num_sampled_nlabel, random_state=random_state)
    concated = pd.concat([df_plabel, sampled_nlabel], axis='index')
    return concated
POSITIVE_LABEL= 1
NEGATIVE_LABEL=0
def bq_gasessions_default(df, columns=[]):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be pandas.DataFrame')
    df = df * 1 
    df = df.fillna(0)

    # drop messy data
    if df.get("clientId") is not None:
        df = drop_messy_data(df, 'clientId')

    if len(columns) > 0:
        df.drop(columns, axis=1, inplace=True)
    return df

def drop_messy_data(df, userlist_key):
    # clientId=='0', visitStartTime=0 を除外
    return df[(~(df[userlist_key]=='0')) & (~(df[userlist_key]==0)) & (~(df['visitStartTime']==0))]


# import google.datalab.storage as storage
from io import BytesIO
from google.cloud import storage as gcs
import re
import pandas as pd
import pickle
PROJECT_NAME = "ea-grothen"
def _open(gcs_path,project_name=None,return_bucket=False):
    # 各名前を取得
    project_name = PROJECT_NAME if project_name is None else project_name
    extension_name =re.search(r"(\..+)",gcs_path).group()
    bucket_name = re.search(r"gs://(.+?)/",gcs_path).group()
    object_name = re.sub(bucket_name,'',gcs_path)
    bucket_name =re.split(r"//(.+?)/",bucket_name)[1]

    client = gcs.Client(project_name)
    bucket = client.get_bucket(bucket_name)
    blob = gcs.Blob(object_name, bucket)
    content = blob.download_as_string()
#     bucket = storage.Bucket(bucket_name)
#     gcs_object = bucket.object(object_name).read_stream()
    
    if extension_name ==".pkl":
        result = pickle.loads(content)
    elif extension_name == ".csv":
        result = pd.read_csv(BytesIO(content))
    if return_bucket:
        return  result, bucket
    return result

from sklearn.utils import check_random_state
from boruta import BorutaPy
import numpy
class BorutaPyForLGB(BorutaPy):
#     lgb_model, n_estimators='auto',n_features_=70, two_step=False,verbose=2,alpha=0.05, random_state=42
    def __init__(self, estimator, n_estimators=1000, perc=100, alpha=0.05,
                 two_step=True, max_iter=100, random_state=None,n_features_=70, verbose=0):
        params=[estimator, n_estimators, perc, alpha,
                         two_step, max_iter, random_state, verbose,n_features_]
        print(len(params))
        super().__init__(estimator, n_estimators=n_estimators,perc =perc,max_iter=max_iter,
                         two_step=two_step,verbose=verbose,alpha=alpha, random_state=random_state)
        if random_state is None:
            self.random_state_input = np.random.randint(0, 2**64-1)
        elif isinstance(random_state, int):
            self.random_state_input = random_state
        else:
            raise TypeError('random_state must be int or None')

    def _get_tree_num(self, n_feat):
        depth = self.estimator.get_params()['max_depth']
        if (depth == None) or (depth <= 0):
            depth = 10
        f_repr = 100
        multi = ((n_feat * 2) / (np.sqrt(n_feat * 2) * depth))
        n_estimators = int(multi * f_repr)
        return n_estimators

    def _fit(self, X, y):
        # check input params
        self._check_params(X, y)
        self.random_state = check_random_state(self.random_state)
        # setup variables for Boruta
        n_sample, n_feat = X.shape
        _iter = 1
        # holds the decision about each feature:
        # 0  - default state = tentative in original code
        # 1  - accepted in original code
        # -1 - rejected in original code
        dec_reg = np.zeros(n_feat, dtype=np.int)
        # counts how many times a given feature was more important than
        # the best of the shadow features
        hit_reg = np.zeros(n_feat, dtype=np.int)
        # these record the history of the iterations
        imp_history = np.zeros(n_feat, dtype=np.float)
        sha_max_history = []

        # set n_estimators
        if self.n_estimators != 'auto':
            self.estimator.set_params(n_estimators=self.n_estimators)

        # main feature selection loop
        while np.any(dec_reg == 0) and _iter < self.max_iter:
            # find optimal number of trees and depth
            if self.n_estimators == 'auto':
                # number of features that aren't rejected
                not_rejected = np.where(dec_reg >= 0)[0].shape[0]
                n_tree = self._get_tree_num(not_rejected)
                self.estimator.set_params(n_estimators=n_tree)

            # make sure we start with a new tree in each iteration
            self.estimator.set_params(random_state=self.random_state_input)

            # add shadow attributes, shuffle them and train estimator, get imps
            cur_imp = self._add_shadows_get_imps(X, y, dec_reg)

            # get the threshold of shadow importances we will use for rejection
            imp_sha_max = np.percentile(cur_imp[1], self.perc)

            # record importance history
            sha_max_history.append(imp_sha_max)
            imp_history = np.vstack((imp_history, cur_imp[0]))

            # register which feature is more imp than the max of shadows
            hit_reg = self._assign_hits(hit_reg, cur_imp, imp_sha_max)

            # based on hit_reg we check if a feature is doing better than
            # expected by chance
            dec_reg = self._do_tests(dec_reg, hit_reg, _iter)

            # print out confirmed features
            if self.verbose > 0 and _iter < self.max_iter:
                self._print_results(dec_reg, _iter, 0)
            if _iter < self.max_iter:
                _iter += 1

        # we automatically apply R package's rough fix for tentative ones
        confirmed = np.where(dec_reg == 1)[0]
        tentative = np.where(dec_reg == 0)[0]
        # ignore the first row of zeros
        tentative_median = np.median(imp_history[1:, tentative], axis=0)
        # which tentative to keep
        tentative_confirmed = np.where(tentative_median
                                       > np.median(sha_max_history))[0]
        tentative = tentative[tentative_confirmed]

        # basic result variables
        self.n_features_ = confirmed.shape[0]
        self.support_ = np.zeros(n_feat, dtype=np.bool)
        self.support_[confirmed] = 1
        self.support_weak_ = np.zeros(n_feat, dtype=np.bool)
        self.support_weak_[tentative] = 1

        # ranking, confirmed variables are rank 1
        self.ranking_ = np.ones(n_feat, dtype=np.int)
        # tentative variables are rank 2
        self.ranking_[tentative] = 2
        # selected = confirmed and tentative
        selected = np.hstack((confirmed, tentative))
        # all rejected features are sorted by importance history
        not_selected = np.setdiff1d(np.arange(n_feat), selected)
        # large importance values should rank higher = lower ranks -> *(-1)
        imp_history_rejected = imp_history[1:, not_selected] * -1

        # update rank for not_selected features
        if not_selected.shape[0] > 0:
                # calculate ranks in each iteration, then median of ranks across feats
                iter_ranks = self._nanrankdata(imp_history_rejected, axis=1)
                rank_medians = np.nanmedian(iter_ranks, axis=0)
                ranks = self._nanrankdata(rank_medians, axis=0)

                # set smallest rank to 3 if there are tentative feats
                if tentative.shape[0] > 0:
                    ranks = ranks - np.min(ranks) + 3
                else:
                    # and 2 otherwise
                    ranks = ranks - np.min(ranks) + 2
                self.ranking_[not_selected] = ranks
        else:
            # all are selected, thus we set feature supports to True
            self.support_ = np.ones(n_feat, dtype=np.bool)

        # notify user
        if self.verbose > 0:
            self._print_results(dec_reg, _iter, 1)
        return self

# import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
def get_corr(df,figsize=(25,25),vmax=1.0,vmin=-1.0,title=None,colormap = plt.cm.RdBu):
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title, y=1.05, size=10)
    df_corr = df.corr()
    sns.heatmap(df_corr,vmax=vmax,vmin=vmin,cmap=colormap,annot=True)
    return df_corr
def ps_dump(objs_nameset,file_path="files/"):
    from inspect import currentframe

    for name , obj in objs_nameset.items():
        with open(file_path + name +".pkl","wb") as f:
            pickle.dump(obj,f)
        print("dumping is done :{}".format(name))

from inspect import currentframe
def get_var_names(*args):
    names ={id(v):k for k,v in currentframe().f_back.f_locals.items()}
    return {names.get(id(arg)) : arg for arg in args}

def p_load(pkl_name,path="files/"):
    with open(path +pkl_name+".pkl","rb") as f:
        return pickle.load(f)
def ps_load(pkl_names,path="files/"):
    pkls ={}
    for pkl_name in pkl_names:
        with open(path +pkl_name+".pkl","rb") as f:
            pkls[pkl_name] = pickle.load(f)
        print("loading is done :{}".format(pkl_name))
    return pkls

def ordered_TS(category_name, categories,df,label_name="label"):
    artificial_time = np.random.permutation(df.index)
    agg_df = df.loc[artificial_time].groupby(category_name).agg({label_name: ['cumsum', 'cumcount']})
    ts = agg_df[(label_name, 'cumsum')] / (agg_df[(label_name, 'cumcount')] + 1)
    return ts


from sklearn.model_selection import StratifiedKFold
def holdout_TS(category_name, categories, df,n_splits=3,label_name="label"):
    folds = StratifiedKFold(n_splits=3,shuffle=True, random_state=42)
    ts = pd.Series(np.empty(df.shape[0]), index=df.index)
    agg_df = df.groupby(category_name).agg({label_name: ['sum', 'count']})
    for _, holdout_idx in folds.split(df, df[label_name]):
        holdout_df = df.iloc[holdout_idx]
        holdout_agg_df = holdout_df.groupby(category_name).agg({'label': ['sum', 'count']})
        train_agg_df = agg_df - holdout_agg_df
        oof_ts = holdout_df.apply(lambda row: train_agg_df.loc[row[category_name]][(label_name, 'sum')] 
                                  / train_agg_df.loc[row[category_name]][(label_name, 'count')] , axis=1)
        ts[oof_ts.index] = oof_ts
    return ts
import scipy as sp
import sklearn.base
class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=100):

        self.dimensions = dimensions
        self.perplexity = perplexity
        self.theta = theta
        self.rand_seed = rand_seed
        self.max_iter = max_iter

    def fit_transform(self, X):
        import bhtsne
        return bhtsne.tsne(
            X.astype(sp.float64),
            dimensions=self.dimensions,
            perplexity=self.perplexity,
            theta=self.theta,
            rand_seed=self.rand_seed,
#             max_iter=self.max_iter
        )
SLACK_NOTIFY_ME="https://hooks.slack.com/services/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"
import slackweb
def slack_notify(text="",
    url=SLACK_NOTIFY_ME): 
    
    slack = slackweb.Slack(url=url)
    slack.notify(text=text)
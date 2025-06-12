import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
from scipy.stats.contingency import association
from scipy.spatial.distance import jensenshannon
from collections import Counter

from datagen import get_data, map_variables, get_topics, get_backstory_variables
from utils import load_pickle, get_probability_vectors, aggregate_answer_for_climate_or_economy
from feature_selection import get_demographic_features, get_attitudinal_features, get_moral_features

def cramer_score(y_true, y_pred):
    contingency_table = pd.crosstab(y_pred, y_true)
    return association(contingency_table.to_numpy(), method="cramer", correction=False)


def run_incremental_model(exp_folder, random_state=None):
    data_folder = os.path.join(exp_folder, 'data')
    demo_feats = get_demographic_features()
    attit_feats = get_attitudinal_features()
    moral_feats = get_moral_features()

    variable_groups = [
                    ('demo', demo_feats),
                    ('demo+attit', demo_feats + attit_feats),
                    ('demo+attit+moral', demo_feats + attit_feats + moral_feats),
                    ('attit', attit_feats),
                    ('attit+moral', attit_feats + moral_feats),
                    ('demo+moral', demo_feats + moral_feats),
                    ('moral', moral_feats)
    ]

    topics = sorted(get_topics())
    model_alias = 'RF'
    results = []

    for group_label, feats_list in variable_groups:
        for topic in topics:
            #output_file = os.path.join(exp_folder, f'{model_alias}_{topic}_{group_label}.csv')
            df_train = load_pickle(data_folder, f'df_train_{topic}.pkl')
            df_val = load_pickle(data_folder, f'df_val_{topic}.pkl')
            #print(f'Validation size for topic {topic}:', df_val.shape)

            X_train = df_train[feats_list]
            X_val = df_val[feats_list]
            y_train = df_train[topic]
            y_val = df_val[topic]

            # aggregate answers for climate_change and current_economy, just like for the LLM
            if topic in ['climate_change', 'current_economy']:
                for i in range(len(y_train)):
                    y_train[i] = aggregate_answer_for_climate_or_economy(topic, y_train[i])
              
                for i in range(len(y_val)):
                    y_val[i] = aggregate_answer_for_climate_or_economy(topic, y_val[i])

            model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=2, random_state=random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # weighted f1-score
            f1 = f1_score(y_val, y_pred, average='weighted')

            # Metrics
            acc = (y_pred == y_val).mean()
            contingency_table = pd.crosstab(y_pred, y_val)
            cramer_v = association(contingency_table.to_numpy(), method="cramer", correction=False)

            # jensen-shannon distance 
            tmp_df = pd.DataFrame(dict(actual_answers=y_val, model_answers=y_pred))
            actual, pred = get_probability_vectors(tmp_df, 'actual_answers', 'model_answers')
            js = jensenshannon(actual, pred, base=2)

            results.append(
                dict(model='RF', topic=topic, feats=group_label, acc=acc, f1_score=f1, js=js, cramer_v=cramer_v)
            )

    df = pd.DataFrame(results)
    df = df.sort_values(by=['topic'])
    return df


def run_model(topic, exp_folder, random_state=None):
    data_folder = os.path.join(exp_folder, 'data')
    df_train = load_pickle(data_folder, f'df_train_{topic}.pkl')
    df_val = load_pickle(data_folder, f'df_val_{topic}.pkl')
    #print(f'Validation size for topic {topic}:', df_val.shape)
    
    # X = df.drop(columns=[topic])
    # y = df[topic]
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    X_train = df_train.drop(columns=[topic])
    y_train = df_train[topic]
    X_val = df_val.drop(columns=[topic])
    y_val = df_val[topic]

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    #cramer_scorer = make_scorer(cramer_score)
    #model = RandomForestClassifier(random_state=random_state)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=2, random_state=random_state)
    #model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, random_state=42)

    #grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
    #grid_search.fit(X_train, y_train)
    model.fit(X_train, y_train)
    #model = grid_search.best_estimator_
    #best_params = grid_search.best_params_
    
    y_pred = model.predict(X_val)

    # Evaluate the model
    # print("Accuracy:", accuracy_score(y_val, y_pred))
    # print("Classification Report:\n", classification_report(y_val, y_pred))

    acc = (y_pred == y_val).mean()

    contingency_table = pd.crosstab(y_pred, y_val)
    cramer_v = association(contingency_table.to_numpy(), method="cramer", correction=False)

    # jensen-shannon distance 
    tmp_df = pd.DataFrame(dict(actual_answers=y_val, model_answers=y_pred))
    actual, pred = get_probability_vectors(tmp_df, 'actual_answers', 'model_answers')
    js = jensenshannon(actual, pred, base=2)

    do_importance = False
    if do_importance:
        importances = model.feature_importances_
        feature_names = X_train.columns

        indices = np.argsort(importances)[::-1]
        top_n = len(feature_names)
        plt.figure(figsize=(10, 6))
        plt.title(topic)
        plt.bar(range(top_n), importances[indices[:top_n]], align="center")
        plt.xticks(range(top_n), feature_names[indices[:top_n]], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    return acc, cramer_v, js


def run_all_topics(exp_folder):
    topics = get_topics()
    topics = sorted(topics)
    results = []

    for topic in topics:
        acc, cramer_v, js = run_model(topic, exp_folder)
        results.append({'topic': topic, 'acc': acc, 'cramer_v': cramer_v, 'js': js})

    df = pd.DataFrame(results)
    return df

if __name__ == '__main__':
    #df = run_all_topics('experiment_free_vars')
    df = run_incremental_model('experiment_incremental_vars', random_state=None)
    print(df.to_string(index=False))
import pandas as pd
import numpy as np
import os
from glob import glob
from scipy.spatial.distance import jensenshannon
from scipy.stats.contingency import association
from sklearn.metrics import f1_score as f1_score_fn

from datagen import get_topics
from utils import get_probability_vectors
import random

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_lower_bounds(js_list, f1_list, cramer_list, lower_percentile):
    lower_bound_js = float(np.percentile(np.array(js_list), lower_percentile))
    lower_bound_f1 = float(np.percentile(np.array(f1_list), lower_percentile))

    if cramer_list:
        lower_bound_cramer = float(np.percentile(np.array(cramer_list), lower_percentile))
    else:
        lower_bound_cramer = 0

    return lower_bound_js, lower_bound_f1, lower_bound_cramer


def get_upper_bounds(js_list, f1_list, cramer_list, upper_percentile):
    upper_bound_js = float(np.percentile(np.array(js_list), upper_percentile))
    upper_bound_f1 = float(np.percentile(np.array(f1_list), upper_percentile))

    if cramer_list:
        upper_bound_cramer = float(np.percentile(np.array(cramer_list), upper_percentile))
    else:
        upper_bound_cramer = 0
    return upper_bound_js, upper_bound_f1, upper_bound_cramer


def get_file_metrics(df_dict, file, col, random_state):
    df = df_dict[file]
    df = df.sample(frac=1.0, replace=True, random_state=random_state)

    actual, pred = get_probability_vectors(df, 'actual_answers', col)
    js = jensenshannon(actual, pred, base=2)
    f1_score = f1_score_fn(df['actual_answers'], df[col], average='weighted')
    ct = pd.crosstab(df['actual_answers'], df[col])
    cramer_v = None

    if not col == 'const':
        cramer_v = association(ct.to_numpy(), method="cramer", correction=True)

        if np.isnan(cramer_v):
            cramer_v = None

    return js, f1_score, cramer_v


def run_bootstrap_files(files_llm, files_rf, col_llm='llm_answers', col_rf='RF_answers', iters=1000, confidence_level=0.95):
    assert len(files_llm) == len(files_rf)

    # LLM
    f1_list = []
    js_list = []
    cramer_list = []

    # RF
    RF_f1_list = []
    RF_js_list = []
    RF_cramer_list = []

    diff_js_list = []
    diff_f1_list = []
    diff_cramer_list = []

    files_llm_dict = {file: pd.read_csv(file) for file in files_llm}
    files_rf_dict = {file: pd.read_csv(file) for file in files_rf}

    len1 = [len(df) for df in files_llm_dict.values()]
    len2 = [len(df) for df in files_rf_dict.values()]
    assert(all(l1 == l2 for l1, l2 in zip(len1, len2)))

    for k in range(iters):
        # LLM
        js = 0
        f1_score = 0
        cramer_v = []

        # RF
        RF_js = 0
        RF_f1_score = 0
        RF_cramer_v = []

        for file_llm, file_rf in zip(files_llm, files_rf):
            if col_rf=='RF_answers':
                # llm feat selec vs rf
                assert len(files_llm_dict[file_llm]) == len(files_rf_dict[file_rf])
                fold_var_llm = ''.join(file_llm.split('_')[-2:])
                fold_var_rf = ''.join(file_rf.split('_')[-2:])
                assert fold_var_llm == fold_var_rf
            elif col_llm == col_rf: # llm feat selec x llm single feat baseline
                assert len(files_llm_dict[file_llm]) == len(files_rf_dict[file_rf])
                p1 = ''.join(file_llm.split('_')[-2])
                p2 = ''.join(file_rf.split('_')[-1])[0]                
                assert p1 == p2
            else: # llm vs random const baseline
                assert len(files_llm_dict[file_llm]) == len(files_rf_dict[file_rf])
                p1 = ''.join(file_llm.split('_')[-1][0])
                p2 = ''.join(file_rf.split('_')[-1][0])
                assert p1 == p2

            # pairwise sample df
            RANDOM_STATE = random.randint(1, 1000000)

            # llm
            js_file, f1_score_file, cramer_v_file = get_file_metrics(files_llm_dict, file_llm, col_llm, RANDOM_STATE)
            js+= js_file
            f1_score+= f1_score_file

            if cramer_v_file is not None:
                cramer_v.append(cramer_v_file)

            del js_file
            del f1_score_file
            del cramer_v_file

            # RF
            js_file, f1_score_file, cramer_v_file = get_file_metrics(files_rf_dict, file_rf, col_rf, RANDOM_STATE)
            RF_js+= js_file
            RF_f1_score+= f1_score_file

            if cramer_v_file is not None:
                RF_cramer_v.append(cramer_v_file)

        # llm
        n = len(files_llm)
        js = js/n
        f1_score = f1_score/n
        cramer_v = sum(cramer_v)/len(cramer_v) if cramer_v else None

        js_list.append(js)
        f1_list.append(f1_score)
        if cramer_v:
            cramer_list.append(cramer_v)

        # RF
        RF_js = RF_js/n
        RF_f1_score = RF_f1_score/n
        RF_cramer_v = sum(RF_cramer_v)/len(RF_cramer_v) if RF_cramer_v else None

        RF_js_list.append(RF_js)
        RF_f1_list.append(RF_f1_score)
        if RF_cramer_v:
            RF_cramer_list.append(RF_cramer_v)

        # diffs
        diff_js_list.append(RF_js - js)
        diff_f1_list.append(f1_score - RF_f1_score)

        if cramer_v is not None and RF_cramer_v is not None:
            diff_cramer_list.append(cramer_v - RF_cramer_v)


    alpha = 1.0 - confidence_level
    lower_percentile = (alpha / 2.0) * 100
    upper_percentile = (1.0 - (alpha / 2.0)) * 100

    # LLM
    lower_bound_js, lower_bound_f1, lower_bound_cramer = get_lower_bounds(js_list, f1_list, cramer_list, lower_percentile)
    upper_bound_js, upper_bound_f1, upper_bound_cramer = get_upper_bounds(js_list, f1_list, cramer_list, upper_percentile)
    # means
    mean_js_llm = np.mean(js_list)
    mean_f1_llm = np.mean(f1_list)
    mean_cramer_llm = np.mean(cramer_list) if cramer_list else 0

    # RF
    lower_bound_js_rf, lower_bound_f1_rf, lower_bound_cramer_rf = get_lower_bounds(RF_js_list, RF_f1_list, RF_cramer_list, lower_percentile)
    upper_bound_js_rf, upper_bound_f1_rf, upper_bound_cramer_rf = get_upper_bounds(RF_js_list, RF_f1_list, RF_cramer_list, upper_percentile)

    # Diff
    lower_bound_js_diff, lower_bound_f1_diff, lower_bound_cramer_diff = get_lower_bounds(diff_js_list, diff_f1_list, diff_cramer_list, lower_percentile)
    upper_bound_js_diff, upper_bound_f1_diff, upper_bound_cramer_diff = get_upper_bounds(diff_js_list, diff_f1_list, diff_cramer_list, upper_percentile)
    # means
    mean_js_diff = np.mean(diff_js_list)
    mean_f1_diff = np.mean(diff_f1_list)
    mean_cramer_diff = np.mean(diff_cramer_list) if diff_cramer_list else 0

    ci_llm = dict(js=[lower_bound_js, upper_bound_js], f1_score=[lower_bound_f1, upper_bound_f1], cramer_v=[lower_bound_cramer, upper_bound_cramer])
    ci_rf = dict(js=[lower_bound_js_rf, upper_bound_js_rf], f1_score=[lower_bound_f1_rf, upper_bound_f1_rf], cramer_v=[lower_bound_cramer_rf, upper_bound_cramer_rf])
    ci_diff = dict(js=[lower_bound_js_diff, upper_bound_js_diff], f1_score=[lower_bound_f1_diff, upper_bound_f1_diff], cramer_v=[lower_bound_cramer_diff, upper_bound_cramer_diff])
    return mean_js_llm, mean_f1_llm, mean_cramer_llm, mean_js_diff, mean_f1_diff, mean_cramer_diff, ci_llm, ci_rf, ci_diff
    

# LLM vs RF
def run_bootstrap_llm_rf(folder, llm_prefix, iters):
    topics = get_topics()

    variable_groups = [
                'demo', 
                'demo+attit',
                'demo+attit+moral',
                'attit',
                'attit+moral',
                'demo+moral',
                'moral'
    ]

    results = []
    for topic in topics:
        for var_pool in variable_groups:
            files_llm = glob(os.path.join(folder, f'{llm_prefix}_{topic}_*_{var_pool}.csv'))
            files_RF = glob(os.path.join(folder, f'RF_{topic}_*_{var_pool}.csv'))

            mean_js_llm, mean_f1_llm, mean_cramer_llm, mean_js_diff, mean_f1_diff, mean_cramer_diff, ci_llm, ci_rf, ci_diff = run_bootstrap_files(files_llm, files_RF, col_llm='llm_answers', col_rf='RF_answers', iters=iters)
            r = {
                'topic': topic, 'group': var_pool,
                'mean_js_llm': mean_js_llm, 'mean_f1_llm': mean_f1_llm, 'mean_cramer_llm': mean_cramer_llm,
                'mean_js_diff': mean_js_diff, 'mean_f1_diff': mean_f1_diff, 'mean_cramer_diff': mean_cramer_diff
            }

            for metric in ['js', 'f1_score', 'cramer_v']:
                r['llm_' + metric + '_lower'] = ci_llm[metric][0]
                r['llm_' + metric + '_upper'] = ci_llm[metric][1]

                r['rf_' + metric + '_lower'] = ci_rf[metric][0]
                r['rf_' + metric + '_upper'] = ci_rf[metric][1]

                r['diff_' + metric + '_lower'] = ci_diff[metric][0]
                r['diff_' + metric + '_upper'] = ci_diff[metric][1]

                metric_sig = not (ci_diff[metric][0] < 0 < ci_diff[metric][1])
                r['diff_sig_' + metric] = metric_sig

            results.append(r)

    df = pd.DataFrame(results)
    outfile = f'bootstrap/bootstrap_{folder}_{llm_prefix}_RF_{iters}.csv'
    df.to_csv(outfile, index=False)
    return df

# LLM Pool vs LLM baseline single feature from each pool
def run_bootstrap_llm_baseline(llm_folder, baseline_folder, llm_prefix, iters):
    topics = get_topics()

    variable_groups = [
                'demo', 
                'demo+attit',
                'demo+attit+moral',
                'attit',
                'attit+moral',
                'demo+moral',
                'moral'
    ]

    results = []
    for topic in topics:
        for var_pool in variable_groups:
            files_llm = glob(os.path.join(llm_folder, f'{llm_prefix}_{topic}_*_{var_pool}.csv'))
            files_baseline = glob(os.path.join(baseline_folder, f'{llm_prefix}_{topic}_*.csv'))

            mean_js_llm, mean_f1_llm, mean_cramer_llm, mean_js_diff, mean_f1_diff, mean_cramer_diff, ci_llm, ci_rf, ci_diff = run_bootstrap_files(files_llm, files_baseline, col_llm='llm_answers', col_rf='llm_answers', iters=iters)
            r = {
                'topic': topic, 'group': var_pool,
                'mean_js_llm': mean_js_llm, 'mean_f1_llm': mean_f1_llm, 'mean_cramer_llm': mean_cramer_llm,
                'mean_js_diff': mean_js_diff, 'mean_f1_diff': mean_f1_diff, 'mean_cramer_diff': mean_cramer_diff
                }

            for metric in ['js', 'f1_score', 'cramer_v']:
                r['llm_' + metric + '_lower'] = ci_llm[metric][0]
                r['llm_' + metric + '_upper'] = ci_llm[metric][1]

                r['baseline_' + metric + '_lower'] = ci_rf[metric][0]
                r['baseline_' + metric + '_upper'] = ci_rf[metric][1]

                r['diff_' + metric + '_lower'] = ci_diff[metric][0]
                r['diff_' + metric + '_upper'] = ci_diff[metric][1]

                metric_sig = not (ci_diff[metric][0] < 0 < ci_diff[metric][1])
                r['diff_sig_' + metric] = metric_sig      

            results.append(r)

    df = pd.DataFrame(results)
    outfile = f'bootstrap/bootstrap_{baseline_folder}_{llm_folder}_{llm_prefix}_{iters}.csv'
    df.to_csv(outfile, index=False)
    return df


def run_bootstrap_random_const_baseline(folder, llm_prefix, other_prefix, col_other, iters):
    topics = get_topics()

    results = []
    for topic in topics:
        files_llm = glob(os.path.join(folder, f'{llm_prefix}_{topic}_*.csv'))
        files_baseline = glob(os.path.join(folder, f'{other_prefix}_{topic}_*.csv'))

        mean_js_llm, mean_f1_llm, mean_cramer_llm, mean_js_diff, mean_f1_diff, mean_cramer_diff, ci_llm, ci_rf, ci_diff = run_bootstrap_files(files_llm, files_baseline, col_llm='llm_answers', col_rf=col_other, iters=iters)
        r = {
            'topic': topic,
            'mean_js_llm': mean_js_llm, 'mean_f1_llm': mean_f1_llm, 'mean_cramer_llm': mean_cramer_llm,
            'mean_js_diff': mean_js_diff, 'mean_f1_diff': mean_f1_diff, 'mean_cramer_diff': mean_cramer_diff
        }

        for metric in ['js', 'f1_score', 'cramer_v']:
            r['llm_' + metric + '_lower'] = ci_llm[metric][0]
            r['llm_' + metric + '_upper'] = ci_llm[metric][1]

            r[f'{other_prefix}_' + metric + '_lower'] = ci_rf[metric][0]
            r[f'{other_prefix}_' + metric + '_upper'] = ci_rf[metric][1]

            r['diff_' + metric + '_lower'] = ci_diff[metric][0]
            r['diff_' + metric + '_upper'] = ci_diff[metric][1]

            metric_sig = not (ci_diff[metric][0] < 0 < ci_diff[metric][1])
            r['diff_sig_' + metric] = metric_sig

        results.append(r)

    df = pd.DataFrame(results)
    outfile = f'bootstrap/bootstrap_{folder}_{llm_prefix}_{other_prefix}_{iters}.csv'
    df.to_csv(outfile, index=False)
    return df

        

def get_ci_table_llm_baseline_llm_pool(folder_baseline, bootstrap_file, model_prefix='Gemma3_12B'):
    from main_results import compare_main_baseline
    df_comp = compare_main_baseline(folder_baseline, model_prefix)
    signif_f = lambda v: '*' if v else ''

    df = pd.read_csv(bootstrap_file)
    df['Topic'] = df['topic'].apply(lambda s: ' '.join(w.capitalize() for w in s.split('_')))
    df['group'] = df['group'].apply(lambda s: '+'.join(w[0].upper() for w in s.split('+')))

    df['llm_f1_ci'] =  '[' + df['llm_f1_score_lower'].round(3).astype(str) + ', ' + df['llm_f1_score_upper'].round(3).astype(str) + ']'
    df['diff_f1_ci'] = '[' + df['diff_f1_score_lower'].round(3).astype(str) + ', ' + df['diff_f1_score_upper'].round(3).astype(str) + ']' + df['diff_sig_f1_score'].apply(signif_f)

    df['llm_js_ci'] = '[' + df['llm_js_lower'].round(3).astype(str) + ', ' + df['llm_js_upper'].round(3).astype(str) + ']'
    df['diff_js_ci'] = '[' + df['diff_js_lower'].round(3).astype(str) + ', ' + df['diff_js_upper'].round(3).astype(str) + ']' + df['diff_sig_js'].apply(signif_f)

    df['llm_cramer_v_ci'] = '[' + df['llm_cramer_v_lower'].round(3).astype(str) + ', ' + df['llm_cramer_v_upper'].round(3).astype(str) + ']'
    df['diff_cramer_v_ci'] = '[' + df['diff_cramer_v_lower'].round(3).astype(str) + ', ' + df['diff_cramer_v_upper'].round(3).astype(str) + ']' + df['diff_sig_cramer_v'].apply(signif_f)

    results = []

    for topic in df_comp['Topic'].tolist():
        r = {'Topic': topic}

        df_cols = []
        pool_f1 = df_comp[(df_comp['Topic'] == topic)]['F1-score pool'].item()
        boot_row_f1 = df[(df['group'] == pool_f1) & (df['Topic'] == topic)][['llm_f1_ci', 'diff_f1_ci']]

        pool_js = df_comp[(df_comp['Topic'] == topic)]['JS pool'].item()
        boot_row_js = df[(df['group'] == pool_js) & (df['Topic'] == topic)][['llm_js_ci', 'diff_js_ci']]

        pool_cramer = df_comp[(df_comp['Topic'] == topic)]['Cramér’s V pool'].item()
        boot_row_cramer = df[(df['group'] == pool_cramer) & (df['Topic'] == topic)][['llm_cramer_v_ci', 'diff_cramer_v_ci']]

        r['F1-score'] = boot_row_f1['llm_f1_ci'].item()
        r['JS'] = boot_row_js['llm_js_ci'].item()
        r['Cramér’s V'] = boot_row_cramer['llm_cramer_v_ci'].item()
        r['Gain F1-score'] = boot_row_f1['diff_f1_ci'].item()
        r['Gain JS'] = boot_row_js['diff_js_ci'].item()
        r['Gain Cramér’s V'] = boot_row_cramer['diff_cramer_v_ci'].item()
        results.append(r)
        
    df = pd.DataFrame(results)
    return df[['Topic', 'F1-score', 'JS', 'Cramér’s V']], df[['Topic', 'Gain F1-score', 'Gain JS', 'Gain Cramér’s V']]


def ci_table_random():
    df = pd.read_csv("bootstrap/bootstrap_experiment_baseline_Gemma3_12B_random_1000.csv")
    df = df.sort_values(by='topic')
    df['topic'] = df['topic'].apply(lambda s: ' '.join(w.capitalize() for w in s.split('_')))

    print("LLM baseline")
    df_res = pd.DataFrame()
    df_res['Topic'] = df['topic'].copy()
    df_res['F1-score'] = '[' + df['llm_f1_score_lower'].round(3).astype(str) + ', ' + df['llm_f1_score_upper'].round(3).astype(str) + ']'
    df_res['JS'] = '[' + df['llm_js_lower'].round(3).astype(str) + ', ' + df['llm_js_upper'].round(3).astype(str) + ']'
    df_res['Cramér’s V'] = '[' + df['llm_cramer_v_lower'].round(3).astype(str) + ', ' + df['llm_cramer_v_upper'].round(3).astype(str) + ']'
    print(df_res.to_latex(index=False, escape=True, column_format='lllllll'))

    print("Diff LLM vs Rand")
    df_res = pd.DataFrame()
    df_res['Topic'] = df['topic'].copy()
    df_res['Gain F1-score'] = '[' + df['diff_f1_score_lower'].round(3).astype(str) + ', ' + df['diff_f1_score_upper'].round(3).astype(str) + ']'
    df_res['Gain JS'] = '[' + df['diff_js_lower'].round(3).astype(str) + ', ' + df['diff_js_upper'].round(3).astype(str) + ']'
    df_res['Gain Cramér’s V'] = '[' + df['diff_cramer_v_lower'].round(3).astype(str) + ', ' + df['diff_cramer_v_upper'].round(3).astype(str) + ']'
    df_res['Topic'] = df_res['Topic'].apply(lambda s: ' '.join(w.capitalize() for w in s.split('_')))
    print(df_res.to_latex(index=False, escape=True, column_format='lllllll'))


def ci_table_const():
    df = pd.read_csv("bootstrap/bootstrap_experiment_baseline_Gemma3_12B_const_1000.csv")
    df = df.sort_values(by='topic')
    df['topic'] = df['topic'].apply(lambda s: ' '.join(w.capitalize() for w in s.split('_')))

    print("Diff LLM vs Const")
    df_res = pd.DataFrame()
    df_res['Topic'] = df['topic'].copy()
    df_res['Gain F1-score'] = '[' + df['diff_f1_score_lower'].round(3).astype(str) + ', ' + df['diff_f1_score_upper'].round(3).astype(str) + ']'
    df_res['Gain JS'] = '[' + df['diff_js_lower'].round(3).astype(str) + ', ' + df['diff_js_upper'].round(3).astype(str) + ']'
    df_res['Gain Cramér’s V'] = '[' + df['diff_cramer_v_lower'].round(3).astype(str) + ', ' + df['diff_cramer_v_upper'].round(3).astype(str) + ']'
    df_res['Topic'] = df_res['Topic'].apply(lambda s: ' '.join(w.capitalize() for w in s.split('_')))
    print(df_res.to_latex(index=False, escape=True, column_format='lllllll'))
    
  

def CI_GAIN_BEST_LLM_RF(folder, llm_pool_map, rf_pool_map, iters, confidence_level=0.95):
    topics = sorted(get_topics())
    alpha = 1.0 - confidence_level
    lower_percentile = (alpha / 2.0) * 100
    upper_percentile = (1.0 - (alpha / 2.0)) * 100
    results = []

    for topic in topics:
        # llm_pool = llm_pool_map[topic]
        # rf_pool = rf_pool_map[topic]

        # file0_llm = os.path.join(folder, f'Gemma3_12B_{topic}_T0.3_0_{llm_pool}.csv')
        # file1_llm = os.path.join(folder, f'Gemma3_12B_{topic}_T0.3_1_{llm_pool}.csv')
        # file2_llm = os.path.join(folder, f'Gemma3_12B_{topic}_T0.3_2_{llm_pool}.csv')

        # file0_rf = os.path.join(folder, f'RF_{topic}_0_{llm_pool}.csv')
        # file1_rf = os.path.join(folder, f'RF_{topic}_1_{llm_pool}.csv')
        # file2_rf = os.path.join(folder, f'RF_{topic}_2_{llm_pool}.csv')

        # llm_df_dict = {f: df.read_csv(f) for f in [file0_llm, file1_llm, file2_llm]}
        # rf_df_dict = {f: df.read_csv(f) for f in [file0_rf, file1_rf, file2_rf]}
        llm_df_dict = {}
        rf_df_dict = {}

        js_llm_list = []
        f1_llm_list = []
        cramer_llm_list = []

        js_rf_list = []
        f1_rf_list = []
        cramer_rf_list = []

        diff_js_list = []
        diff_f1_list = []
        diff_cramer_list = []

        for i in range(iters):
            RANDOM_STATE = random.randint(1, 1000000)

            fold_scores_llm = {'js':[], 'cramer': [], 'f1': []}
            for metric in fold_scores_llm.keys():
                llm_pool = llm_pool_map[metric][topic]
                file0_llm = os.path.join(folder, f'Gemma3_12B_{topic}_T0.3_0_{llm_pool}.csv')
                file1_llm = os.path.join(folder, f'Gemma3_12B_{topic}_T0.3_1_{llm_pool}.csv')
                file2_llm = os.path.join(folder, f'Gemma3_12B_{topic}_T0.3_2_{llm_pool}.csv')


                for f in [file0_llm, file1_llm, file2_llm]:
                    if f not in llm_df_dict:
                        llm_df_dict[f] = pd.read_csv(f)

                    js_llm, f1_llm, cramer_v_llm = get_file_metrics(llm_df_dict, f, 'llm_answers', RANDOM_STATE)
                    metrics = {'js': js_llm, 'f1': f1_llm, 'cramer': cramer_v_llm}

                    if metrics[metric]:
                        fold_scores_llm[metric].append(metrics[metric])

            fold_scores_rf = {'js':[], 'cramer': [], 'f1': []}
            for metric in fold_scores_rf.keys():
                rf_pool = rf_pool_map[metric][topic]
                file0_rf = os.path.join(folder, f'RF_{topic}_0_{rf_pool}.csv')
                file1_rf = os.path.join(folder, f'RF_{topic}_1_{rf_pool}.csv')
                file2_rf = os.path.join(folder, f'RF_{topic}_2_{rf_pool}.csv')


                for f in [file0_rf, file1_rf, file2_rf]:
                    if f not in rf_df_dict:
                        rf_df_dict[f] = pd.read_csv(f)

                    js_rf, f1_rf, cramer_v_rf = get_file_metrics(rf_df_dict, f, 'RF_answers', RANDOM_STATE)
                    metrics = {'js': js_rf, 'f1': f1_rf, 'cramer': cramer_v_rf}

                    if metrics[metric]:
                        fold_scores_rf[metric].append(metrics[metric])


            js_llm_list.append(np.mean(fold_scores_llm['js']))
            f1_llm_list.append(np.mean(fold_scores_llm['f1']))
            cramer_llm_list.append(np.mean(fold_scores_llm['cramer']))

            js_rf_list.append(np.mean(fold_scores_rf['js']))
            f1_rf_list.append(np.mean(fold_scores_rf['f1']))
            cramer_rf_list.append(np.mean(fold_scores_rf['cramer']))

            diff_js_list.append(js_rf_list[-1] - js_llm_list[-1])
            diff_f1_list.append(f1_llm_list[-1] - f1_rf_list[-1])
            diff_cramer_list.append(cramer_llm_list[-1] - cramer_rf_list[-1])


        # LLM
        lower_bound_js, lower_bound_f1, lower_bound_cramer = get_lower_bounds(js_llm_list, f1_llm_list, cramer_llm_list, lower_percentile)
        upper_bound_js, upper_bound_f1, upper_bound_cramer = get_upper_bounds(js_llm_list, f1_llm_list, cramer_llm_list, upper_percentile)

        # RF
        lower_bound_js_rf, lower_bound_f1_rf, lower_bound_cramer_rf = get_lower_bounds(js_rf_list, f1_rf_list, cramer_rf_list, lower_percentile)
        upper_bound_js_rf, upper_bound_f1_rf, upper_bound_cramer_rf = get_upper_bounds(js_rf_list, f1_rf_list, cramer_rf_list, upper_percentile)

        # Diff
        lower_bound_js_diff, lower_bound_f1_diff, lower_bound_cramer_diff = get_lower_bounds(diff_js_list, diff_f1_list, diff_cramer_list, lower_percentile)
        upper_bound_js_diff, upper_bound_f1_diff, upper_bound_cramer_diff = get_upper_bounds(diff_js_list, diff_f1_list, diff_cramer_list, upper_percentile)

        ci_llm = dict(js=[lower_bound_js, upper_bound_js], f1_score=[lower_bound_f1, upper_bound_f1], cramer_v=[lower_bound_cramer, upper_bound_cramer])
        ci_rf = dict(js=[lower_bound_js_rf, upper_bound_js_rf], f1_score=[lower_bound_f1_rf, upper_bound_f1_rf], cramer_v=[lower_bound_cramer_rf, upper_bound_cramer_rf])
        ci_diff = dict(js=[lower_bound_js_diff, upper_bound_js_diff], f1_score=[lower_bound_f1_diff, upper_bound_f1_diff], cramer_v=[lower_bound_cramer_diff, upper_bound_cramer_diff])

        result_row = {
            'Topic': ' '.join(w.capitalize() for w in topic.split('_')),
            'Gain F1-score': '[' + str(round(ci_diff['f1_score'][0], 3)) + ', ' + str(round(ci_diff['f1_score'][1], 3)) + ']',
            'Gain JS': '[' + str(round(ci_diff['js'][0], 3)) + ', ' + str(round(ci_diff['js'][1], 3)) + ']',
            'Gain Cramér’s V': '[' + str(round(ci_diff['cramer_v'][0], 3)) + ', ' + str(round(ci_diff['cramer_v'][1], 3)) + ']'
        }
        results.append(result_row)

    return pd.DataFrame(results)


def exp2_no_selection_best_pools():
    llm_pool_map = {
        'cramer': {
            'climate_change': 'attit+moral',
            'current_economy': 'attit+moral',
            'drug_addiction': 'moral',
            'gay_marriage': 'attit',
            'gender_role': 'demo+attit',
            'gun_regulation': 'demo+attit',
            'health_insurance_policy': 'moral',
            'income_inequality': 'attit+moral',
            'race_diversity': 'demo+attit',
            'refugee_allowing': 'demo+attit+moral'
        },
        'js': {
            'climate_change': 'attit+moral',
            'current_economy': 'attit',
            'drug_addiction': 'attit',
            'gay_marriage': 'attit',
            'gender_role': 'attit',
            'gun_regulation': 'attit',
            'health_insurance_policy': 'demo+attit',
            'income_inequality': 'attit',
            'race_diversity': 'attit',
            'refugee_allowing': 'attit'
        },
        'f1': {
            'climate_change': 'attit+moral',
            'current_economy': 'attit',
            'drug_addiction': 'moral',
            'gay_marriage': 'attit', 
            'gender_role': 'demo+attit+moral',
            'gun_regulation': 'attit',
            'health_insurance_policy': 'demo+attit+moral',
            'income_inequality': 'attit+moral',
            'race_diversity': 'attit',
            'refugee_allowing': 'attit'
        }
    }
    rf_pool_map = {
        'cramer': {
            'climate_change': 'attit+moral',
            'current_economy': 'attit+moral',
            'drug_addiction': 'demo+attit+moral',
            'gay_marriage': 'attit+moral',
            'gender_role': 'demo+attit+moral',
            'gun_regulation': 'demo+attit+moral',
            'health_insurance_policy': 'attit+moral',
            'income_inequality': 'attit+moral',
            'race_diversity': 'demo+attit+moral',
            'refugee_allowing': 'attit+moral'
        },
        'js': {
            'climate_change': 'moral',
            'current_economy': 'moral',
            'drug_addiction': 'moral',
            'gay_marriage': 'attit',
            'gender_role': 'attit',
            'gun_regulation': 'moral',
            'health_insurance_policy': 'moral',
            'income_inequality': 'attit',
            'race_diversity': 'moral',
            'refugee_allowing': 'moral'
        },
        'f1': {
            'climate_change': 'attit+moral',
            'current_economy': 'attit+moral',
            'drug_addiction': 'attit+moral',
            'gay_marriage': 'demo+attit',
            'gender_role': 'attit+moral',
            'gun_regulation': 'demo+attit+moral',
            'health_insurance_policy': 'attit+moral',
            'income_inequality': 'attit+moral',
            'race_diversity': 'demo+attit+moral',
            'refugee_allowing': 'attit+moral'
        }
    }
    return llm_pool_map, rf_pool_map


def exp3_selection_best_pools():
    llm_pool_map = {
        'cramer': {
            'climate_change': 'demo+attit+moral',
            'current_economy': 'attit',
            'drug_addiction': 'demo+moral',
            'gay_marriage': 'demo+attit',
            'gender_role': 'demo+attit+moral',
            'gun_regulation': 'attit',
            'health_insurance_policy': 'demo+moral',
            'income_inequality': 'attit',
            'race_diversity': 'demo+attit',
            'refugee_allowing': 'demo+attit'
        },
        'js': {
            'climate_change': 'demo+attit',
            'current_economy': 'attit',
            'drug_addiction': 'demo+attit',
            'gay_marriage': 'attit',
            'gender_role': 'demo+attit+moral',
            'gun_regulation': 'attit+moral',
            'health_insurance_policy': 'attit+moral',
            'income_inequality': 'attit',
            'race_diversity': 'demo+attit',
            'refugee_allowing': 'demo'
        },
        'f1': {
            'climate_change': 'demo+attit',
            'current_economy': 'attit+moral',
            'drug_addiction': 'demo+moral',
            'gay_marriage': 'demo+attit+moral', 
            'gender_role': 'demo+attit+moral',
            'gun_regulation': 'demo+attit',
            'health_insurance_policy': 'demo+moral',
            'income_inequality': 'attit',
            'race_diversity': 'demo+attit',
            'refugee_allowing': 'demo+attit'
        }
    }
    rf_pool_map = {
        'cramer': {
            'climate_change': 'attit+moral',
            'current_economy': 'attit+moral',
            'drug_addiction': 'demo+attit+moral',
            'gay_marriage': 'attit+moral',
            'gender_role': 'demo+attit+moral',
            'gun_regulation': 'demo+attit+moral',
            'health_insurance_policy': 'attit+moral',
            'income_inequality': 'attit+moral',
            'race_diversity': 'demo+attit+moral',
            'refugee_allowing': 'attit+moral'
        },
        'js': {
            'climate_change': 'moral',
            'current_economy': 'moral',
            'drug_addiction': 'moral',
            'gay_marriage': 'attit',
            'gender_role': 'attit',
            'gun_regulation': 'moral',
            'health_insurance_policy': 'moral',
            'income_inequality': 'attit',
            'race_diversity': 'moral',
            'refugee_allowing': 'moral'
        },
        'f1': {
            'climate_change': 'attit+moral',
            'current_economy': 'attit+moral',
            'drug_addiction': 'attit+moral',
            'gay_marriage': 'demo+attit',
            'gender_role': 'attit+moral',
            'gun_regulation': 'demo+attit+moral',
            'health_insurance_policy': 'attit+moral',
            'income_inequality': 'attit+moral',
            'race_diversity': 'demo+attit+moral',
            'refugee_allowing': 'attit+moral'
        }
    }
    return llm_pool_map, rf_pool_map


def get_ci_table_llm_rf_exp2(iters=4):
    llm_pool_map, rf_pool_map = exp2_no_selection_best_pools()
    df = CI_GAIN_BEST_LLM_RF('experiment_all_folds_no_selection', llm_pool_map, rf_pool_map, iters)
    print(df.to_latex(index=False, escape=True, column_format='lllllll'))


def get_ci_table_llm_rf_exp3(iters=4):
    llm_pool_map, rf_pool_map = exp3_selection_best_pools()
    df = CI_GAIN_BEST_LLM_RF('experiment_all_folds', llm_pool_map, rf_pool_map, iters)
    print(df.to_latex(index=False, escape=True, column_format='lllllll'))

if __name__ == '__main__':
    # ITERS = 1000
    # df = run_bootstrap_llm_rf('experiment_all_folds', 'Gemma3_12B', ITERS)
    # df = run_bootstrap_llm_rf('experiment_all_folds_no_selection', 'Gemma3_12B', ITERS)
    # df = run_bootstrap_llm_baseline('experiment_all_folds', 'experiment_baseline', 'Gemma3_12B', ITERS)
    # df = run_bootstrap_llm_baseline('experiment_all_folds_no_selection', 'experiment_baseline', 'Gemma3_12B', ITERS)
    # df = run_bootstrap_random_const_baseline('experiment_baseline', 'Gemma3_12B', 'random', 'random', ITERS)
    # df = run_bootstrap_random_const_baseline('experiment_baseline', 'Gemma3_12B', 'const', 'const', ITERS)

    # Experiment 2
    # df1, df2 = get_ci_table_llm_baseline_llm_pool('experiment_all_folds_no_selection', 'bootstrap/bootstrap_experiment_baseline_experiment_all_folds_no_selection_Gemma3_12B_1000.csv')
    # print(df1.to_latex(index=False, escape=True, column_format='lllllll'))
    # print(df2.to_latex(index=False, escape=True, column_format='lllllll'))


    # Experiment 3
    # df1, df2 = get_ci_table_llm_baseline_llm_pool('experiment_all_folds', 'bootstrap/bootstrap_experiment_baseline_experiment_all_folds_Gemma3_12B_1000.csv')
    # print(df1.to_latex(index=False, escape=True, column_format='lllllll'))
   
    # print("RF CI EXP 2")
    # get_ci_table_llm_rf_exp2(iters=1000)

    # print("\nRF CI EXP 3")
    # get_ci_table_llm_rf_exp3(iters=1000)

    # ci_table_random()
    ci_table_const()
    

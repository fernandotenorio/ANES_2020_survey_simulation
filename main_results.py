from utils import get_probability_vectors
from glob import glob
import pandas as pd
import os
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association
from sklearn.metrics import f1_score as f1_score_fn
import numpy as np
from datagen import get_topics
from utils import load_pickle, aggregate_answer_for_climate_or_economy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib
import re


def is_dark_color(r, g, b):
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 150

def formatter_cell(val):
        cmap = matplotlib.colormaps['RdBu']
        vmin, vmax = -0.3, 0.3
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        r, g, b, _ = cmap(norm(val))
        r255, g255, b255 = int(r * 255), int(g * 255), int(b * 255)
        cell_color = f"\\cellcolor[RGB]{{{r255},{g255},{b255}}}"
        text_color = "\\textcolor{white}" if is_dark_color(r255, g255, b255) else ""
        return f"{cell_color}{text_color}{{{val:.2f}}}"

def _get_metrics(files, col):
    js = 0
    f1_score = 0
    cramer_v = 0
    # min_p_value = float('+inf')
    # max_p_value = float('-inf')

    for file in files:
        df = pd.read_csv(file)
        actual, pred = get_probability_vectors(df, 'actual_answers', col)
        js+= jensenshannon(actual, pred, base=2)
        f1_score += f1_score_fn(df['actual_answers'], df[col], average='weighted')
        ct = pd.crosstab(df['actual_answers'], df[col])        
        # _, p_value, _, _ = chi2_contingency(ct)

        if not col == 'const':
            cramer_v+= association(ct.to_numpy(), method="cramer", correction=True)

        # if p_value < min_p_value:
        #     min_p_value = p_value
        # if p_value > max_p_value:
        #     max_p_value = p_value

    n =  len(files)
    js = js/n
    f1_score = f1_score/n
    cramer_v = cramer_v/n
    return dict(js=js, f1_score=f1_score, cramer_v=cramer_v)


def run(folder, model_prefix):
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
            files_llm = glob(os.path.join(folder, f'{model_prefix}_{topic}_*_{var_pool}.csv'))
            files_RF = glob(os.path.join(folder, f'RF_{topic}_*_{var_pool}.csv'))
            files_gpt4o = glob(os.path.join(folder, f'ChatGPT-4o_{topic}_*_{var_pool}.csv'))

            llm_model_metrics = _get_metrics(files_llm, 'llm_answers')
            rf_model_metrics = _get_metrics(files_RF, 'RF_answers')

            if files_gpt4o:
                gpt4_metrics = _get_metrics(files_gpt4o, 'llm_answers')
            else:
                gpt4_metrics = {}

            var_pool = '+'.join(v.capitalize() for v in var_pool.split('+'))
            row = {'topic': topic, 'var_pool': var_pool}

            for metric, val in llm_model_metrics.items():
                row[metric] = val

            for metric, val in rf_model_metrics.items():
                row['rf_' + metric] = val

            for metric, val in gpt4_metrics.items():
                row['gpt4o_' + metric] = val


            for metric in llm_model_metrics.keys(): 
                llm_val = llm_model_metrics[metric]
                rf_val = rf_model_metrics[metric]

                if metric == 'js':
                    diff = rf_val - llm_val                 
                else:
                    diff = llm_val - rf_val
                row[metric + '_diff'] = diff

            # gpt4o
            for metric in gpt4_metrics.keys(): 
                llm_val = gpt4_metrics[metric]
                rf_val = rf_model_metrics[metric]

                if metric == 'js':
                    diff = rf_val - llm_val                 
                else:
                    diff = llm_val - rf_val
                row[metric + '_diff'] = diff
        
            results.append(row)
    return pd.DataFrame(results)



def heatmap(df, filename):
    # topic_to_plot = topic
    cols = [c for c in df.columns if c in ['f1_score_diff', 'cramer_v_diff', 'js_diff']]
    df = df[cols]

    df_for_coloring = df.copy()
    red_white_green = LinearSegmentedColormap.from_list("RwG", ["red", "white", "green"])
    
    col_labels = {'js_diff': 'JS Distance', 'f1_score_diff': 'F1-score', 'cramer_v_diff': 'Cramér’s V'}
    df_for_coloring = df_for_coloring.rename(columns=col_labels)

    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(
        df_for_coloring,   
        center=0,
        annot=df, 
        fmt=".2f",    
        cmap='RdBu',  
        linecolor='white',
        linewidths=0.5,
        cbar=False
    )

    # Move column labels (x-axis ticks and label) to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # Optional: Rotate x-axis tick labels if they are long
    ax.tick_params(axis='x', labelrotation=0) # Set to 0 for horizontal, or e.g. 45 for rotated
    #plt.title(f"Performance Metrics for Topic: {topic_to_plot}", pad=40) # Add padding for top labels
    plt.title(None)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap, may need tweaking
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()


def generate_latex_table(df, caption, label):
    table = []
    table.append(r"\begin{table}[ht]")
    table.append(r"\centering")
    #table.append(r"\caption{Performance comparison between LLM and RF models. Reported metrics are averaged across folds.}")
    table.append(f"\\caption{{{caption}}}")
    table.append(r"\begin{tabular}{llccc}")
    table.append(r"\toprule")
    table.append(r"\textbf{Variable Pool} & \textbf{Model} & \textbf{F1-score} & \textbf{JS Distance} & \textbf{Cramér’s V} \\")
    table.append(r"\midrule")

    for var in df['var_pool']:
        sub_df = df[df['var_pool'] == var]
        f1_llm = sub_df['f1_score'].values[0]
        f1_rf = sub_df['rf_f1_score'].values[0]
        f1_llm_text = f'{f1_llm:.2f}'
        f1_rf_text = f'{f1_rf:.2f}'

        js_llm = sub_df['js'].values[0]
        js_rf = sub_df['rf_js'].values[0]
        js_llm_text = f'{js_llm:.2f}'
        js_rf_text = f'{js_rf:.2f}'

        cramer_llm = sub_df['cramer_v'].values[0]
        cramer_rf = sub_df['rf_cramer_v'].values[0]
        cramer_llm_text = f'{cramer_llm:.2f}'
        cramer_rf_text = f'{cramer_rf:.2f}'

        # gpt4o
        gpt4o_f1 = sub_df['gpt4o_f1_score'].values[0]
        gpt4o_f1_text = f'{gpt4o_f1:.2f}'

        gpt4o_js = sub_df['gpt4o_js'].values[0]
        gpt4o_js_text = f'{gpt4o_js:.2f}'

        gpt4o_cramer_v = sub_df['gpt4o_cramer_v'].values[0]
        gpt4o_cramer_v_text = f'{gpt4o_cramer_v:.2f}'

        table.append(f"{var.title()}     & Gemma3 12B      & {f1_llm_text}  & {js_llm_text} & {cramer_llm_text} \\\\")
        table.append(f"                   & ChatGPT-4o   & {gpt4o_f1_text}  & {gpt4o_js_text} & {gpt4o_cramer_v_text} \\\\")
        table.append(f"                   & RF   & {f1_rf_text}  & {js_rf_text} & {cramer_rf_text} \\\\")
        table.append(r"\addlinespace")

    table.append(r"\bottomrule")
    table.append(r"\end{tabular}")
    # table.append(r"\label{tab:model_comparison}")
    table.append(f"\\label{{{label}}}")
    table.append(r"\end{table}")

    return '\n'.join(table)


def heatmap_table(diff_df, caption="Some caption", label='Some Label'):
    cmap = matplotlib.colormaps['RdBu']
    vmin, vmax = -0.3, 0.3
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cols = [c for c in diff_df.columns if c in ['f1_score_diff', 'cramer_v_diff', 'js_diff']]
    diff_df = diff_df[cols]

    col_labels = {'js_diff': 'JS Distance', 'f1_score_diff': 'F1-score', 'cramer_v_diff': 'Cramér’s V'}
    diff_df = diff_df.rename(columns=col_labels)

    latex_rows = []
    for idx, row in diff_df.iterrows():
        latex_row = [idx]
        for val in row:
            r, g, b, _ = cmap(norm(val))
            r255, g255, b255 = int(r * 255), int(g * 255), int(b * 255)
            cell_color = f"\\cellcolor[RGB]{{{r255},{g255},{b255}}}"
            text_color = "\\textcolor{white}" if is_dark_color(r255, g255, b255) else ""
            latex_row.append(f"{cell_color}{text_color}{{{val:.2f}}}")
        latex_rows.append(" & ".join(latex_row) + r" \\")

    latex_table_with_textcolor = r"""\begin{table}[ht]""" + "\n"
    latex_table_with_textcolor+= r"""\centering""" + "\n"
    latex_table_with_textcolor += r"""\begin{tabular}{l""" + "c" * len(diff_df.columns) + "}\n"
    latex_table_with_textcolor += "\\toprule\n"
    #latex_table_with_textcolor += "Variable Pool & " + " & ".join(diff_df.columns) + r" \\" + '\n'
    latex_table_with_textcolor += "\\textbf{Variable Pool} & " + " & ".join([f"\\textbf{{{col}}}" for col in diff_df.columns]) + r" \\" + '\n'
    latex_table_with_textcolor += "\\midrule\n"
    latex_table_with_textcolor += "\n".join(latex_rows)
    latex_table_with_textcolor += "\\addlinespace\n"
    latex_table_with_textcolor += "\\bottomrule\n"
    latex_table_with_textcolor += "\\end{tabular}\n"
    latex_table_with_textcolor+= f"\\caption{{{caption}}}\n"
    latex_table_with_textcolor+= f"\\label{{{label}}}\n"
    latex_table_with_textcolor+= r"\end{table}"
    return latex_table_with_textcolor


def var_pool_avg():
    df = run('experiment_all_folds', 'Gemma3_12B')

    summary_avg = df.groupby('var_pool').agg(
        f1_score=('f1_score', 'mean'),
        rf_f1_score=('rf_f1_score', 'mean'),
        gpt4o_f1_score=('gpt4o_f1_score', 'mean'),
        js=('js', 'mean'),
        rf_js=('rf_js', 'mean'),
        gpt4o_js=('gpt4o_js', 'mean'),
        cramer_v=('cramer_v', 'mean'),
        rf_cramer_v=('rf_cramer_v', 'mean'),
        gpt4o_cramer_v=('gpt4o_cramer_v', 'mean'),
    ).reset_index()

    print(summary_avg)

    
    print(generate_latex_table(summary_avg, 'Average Performance Metrics Across All Topics by Variable Pool', 'tab:acc_pool'))
    summary_avg.set_index('var_pool', inplace=True)
    summary_avg['f1_score_diff'] = summary_avg['f1_score'] - summary_avg['rf_f1_score']
    summary_avg['cramer_v_diff'] = summary_avg['cramer_v'] - summary_avg['rf_cramer_v']
    summary_avg['js_diff'] = summary_avg['rf_js'] - summary_avg['js']
    #heatmap(summary_avg, "tex_src/images/acc_pool_heatmap_gemma.png")
    print("Gemma heatmap")
    print(heatmap_table(summary_avg, 'Performance difference [Gemma3 12B - RF]', 'tab:acc_pool_gemma'))

    summary_avg['f1_score_diff'] = summary_avg['gpt4o_f1_score'] - summary_avg['rf_f1_score']
    summary_avg['cramer_v_diff'] = summary_avg['gpt4o_cramer_v'] - summary_avg['rf_cramer_v']
    summary_avg['js_diff'] = summary_avg['rf_js'] - summary_avg['gpt4o_js']
    print("ChatGPT heatmap")
    #heatmap(summary_avg, "tex_src/images/acc_pool_heatmap_gpt4o.png")
    print(heatmap_table(summary_avg, 'Performance difference [ChatGPT-4o - RF]', 'tab:acc_pool_gpt'))

    summary_avg['f1_score_diff'] = summary_avg['f1_score'] - summary_avg['gpt4o_f1_score']
    summary_avg['cramer_v_diff'] = summary_avg['cramer_v'] - summary_avg['gpt4o_cramer_v']
    summary_avg['js_diff'] = summary_avg['gpt4o_js'] - summary_avg['js']
    print("Gemma - ChatGPT heatmap")
    #heatmap(summary_avg, "tex_src/images/acc_pool_heatmap_gemma_gpt4o.png")
    print(heatmap_table(summary_avg, 'Performance difference [Gemma3 12B - ChatGPT-4o]', 'tab:acc_pool_gpt_gemma'))


# folder: experiment_all_folds_no_selection, experiment_all_folds
def get_pool_rf(folder):
    df = run(folder, 'Gemma3_12B')
    df_res = pd.DataFrame()

    score_cols = ['rf_f1_score', 'rf_js', 'rf_cramer_v']
    #score_cols = ['f1_score', 'js', 'cramer_v']

    for col in score_cols:
        if 'js' in col:
            df_best = df.loc[df.groupby("topic")[col].idxmin(), ["topic", "var_pool", col]]
        else:
            df_best = df.loc[df.groupby("topic")[col].idxmax(), ["topic", "var_pool", col]]

        df_best = df_best.reset_index(drop=True)

        if 'topic' not in df_res.columns:
            df_res['topic'] = df_best['topic'].copy()

        df_res['best_' + col] = df_best[col]
        df_res['best_pool_' + col] = df_best['var_pool'].apply(lambda s: '+'.join(w[0].upper() for w in s.split('+')))


    df_res = df_res[['topic', 'best_rf_f1_score', 'best_rf_js', 'best_rf_cramer_v', 'best_pool_rf_f1_score', 'best_pool_rf_js', 'best_pool_rf_cramer_v']]
    cols = {
            'topic': 'Topic',
            'best_rf_f1_score': 'F1-score',
            'best_pool_rf_f1_score': 'F1-score pool',
            'best_rf_js': 'JS',
            'best_pool_rf_js': 'JS pool',
            'best_rf_cramer_v': 'Cramér’s V',
            'best_pool_rf_cramer_v': 'Cramér’s V pool'
    }
    df_res = df_res.rename(columns=cols)
    df_res['Topic'] = df_res['Topic'].apply(lambda s: ' '.join(w.capitalize() for w in s.split('_')))
    print(df_res.to_latex(index=False, escape=True, column_format='lcccccc', float_format="%.2f"))
    return df_res


# Experiment 2
def compare_main_baseline(folder, model_prefix):
    from baseline_results import run as run_baseline
    df_base = run_baseline('experiment_baseline', 'Gemma3_12B')
    df_base = df_base.sort_values(by='topic')
    df_base = df_base[['topic', 'llm_f1_score', 'llm_js', 'llm_cramer_v']]
    df_base = df_base.reset_index(drop=True)

    df = run(folder, model_prefix)
    score_cols = ['f1_score', 'js', 'cramer_v']

    for col in score_cols:
        if col == 'js':
            df_best = df.loc[df.groupby("topic")[col].idxmin(), ["topic", "var_pool", col]]
        else:
            df_best = df.loc[df.groupby("topic")[col].idxmax(), ["topic", "var_pool", col]]

        df_best = df_best.reset_index(drop=True)
        df_best = df_best.sort_values(by='topic')
        df_base['best_' + col] = df_best[col]
        df_base['best_pool_' + col] = df_best['var_pool'].apply(lambda s: '+'.join(w[0].upper() for w in s.split('+')))
       
        if col == 'js':
            df_base['diff_' + col] = df_base['llm_' + col] - df_best[col]
        else:
            df_base['diff_' + col] = df_best[col] - df_base['llm_' + col]
    
    df_base = df_base[['topic', 'best_f1_score', 'best_js', 'best_cramer_v', 'diff_f1_score', 'diff_js', 'diff_cramer_v', 'best_pool_f1_score', 'best_pool_js', 'best_pool_cramer_v']]
    col_rename = {
                'topic': 'Topic',
                'best_f1_score': 'F1-score', 'best_js': 'JS', 'best_cramer_v': 'Cramér’s V', 'diff_f1_score': 'Gain F1-score', 'diff_js': 'Gain JS', 'diff_cramer_v': 'Gain Cramér’s V',
                'best_pool_f1_score' : 'F1-score pool',
                'best_pool_js': 'JS pool',
                'best_pool_cramer_v' : 'Cramér’s V pool'
    }
    df_base = df_base.rename(columns=col_rename)
    df_base['Topic'] = df_base['Topic'].apply(lambda s: ' '.join(w.capitalize() for w in s.split('_')))

    formatter_cells = {k: formatter_cell for k in df_base.keys() if 'Gain' in k}
    #print(df_base.to_latex(index=False, escape=True, column_format='lccccccccc', float_format="%.2f", formatters=formatter_cells))
    cols_metrics = [c for c in df_base.columns if not ('Gain' in c)]
    cols_gains = ['Topic'] + [c for c in df_base.columns if 'Gain' in c]

    df_point_metrics_pools = df_base[cols_metrics]
    df_gains = df_base[cols_gains]
    print(df_point_metrics_pools.to_latex(index=False, escape=True, column_format='lccccccccc', float_format="%.2f", formatters=formatter_cells))
    print(df_gains.to_latex(index=False, escape=True, column_format='lccccccccc', float_format="%.2f", formatters=formatter_cells))
    return df_base


def compare_rf_llm_pool(folder='experiment_all_folds_no_selection', model_prefix='Gemma3_12B'):
    df = run(folder, model_prefix)
    #df = run('experiment_all_folds_qwen_selection', 'Qwen2.5_14B')

    prefix = ['', 'rf_']
    res_df = pd.DataFrame()

    for p in prefix:
        score_cols = [p + 'f1_score', p + 'js', p + 'cramer_v']

        for col in score_cols:
            if 'js'in col:
                df_best = df.loc[df.groupby("topic")[col].idxmin(), ["topic", "var_pool", col]]
            else:
                df_best = df.loc[df.groupby("topic")[col].idxmax(), ["topic", "var_pool", col]]

            df_best = df_best.reset_index(drop=True)
            df_best = df_best.sort_values(by='topic')
            
            if 'topic' not in res_df.columns:
                res_df['topic'] = df_best['topic'].copy()

            res_df[col] = df_best[col].copy()

    # set diffs
    for col in ['f1_score', 'js', 'cramer_v']:
        # diffs relative to the LLM
        if 'js' == col:
            res_df['diff_' + col] = res_df['rf_' + col] - res_df[col]
        else:
            res_df['diff_' + col] = res_df[col] - res_df['rf_' + col]

    res_df = res_df[['topic', 'rf_f1_score', 'rf_js', 'rf_cramer_v', 'diff_f1_score', 'diff_js', 'diff_cramer_v']]

    # remap cols
    cols_rename = {
        'topic': 'Topic',
        # 'f1_score': 'F1 score (LLM)',
        # 'js': 'JS (LLM)',
        # 'cramer_v': 'Cramér’s V (LLM)',
        # 'rf_f1_score': 'F1-score',
        # 'rf_js': 'JS',
        # 'rf_cramer_v': 'Cramér’s V',
        'diff_f1_score': 'Diff F1-score',
        'diff_js': 'Diff JS',
        'diff_cramer_v': 'Diff Cramér’s V'
    }

    res_df = res_df.rename(columns=cols_rename)    
    res_df['Topic'] = res_df['Topic'].apply(lambda s: ' '.join(w.capitalize() for w in s.split('_')))
    formatter_cells = {k: formatter_cell for k in res_df.keys() if 'Diff' in k}

    show_cols = ['Topic'] + [c for c in res_df.columns if 'Diff' in c]
    print(res_df[show_cols].to_latex(index=False, escape=True, column_format='lccccccccc', float_format="%.2f", formatters=formatter_cells))


def head_head_signif():
    df = pd.read_csv("bootstrap/bootstrap_experiment_all_folds_no_selection_Gemma3_12B_RF_1000.csv")
    topics = sorted(set(df['topic'].tolist()))
    groups = sorted([
                'demo', 
                'demo+attit',
                'demo+attit+moral',
                'attit',
                'attit+moral',
                'demo+moral',
                'moral'
    ])

    group_abbrev = ['+'.join(w[0].upper() for w in g.split('+')) for g in groups]
    topics_name = [' '.join(w.capitalize() for w in t.split('_')) for t in topics]

    data = {}
    metrics = ['f1_score', 'js', 'cramer_v']
    for metric in metrics:
        results = []
        for topic in topics:
            row = []
            for group in groups:
                df_tmp = df[(df['topic'] == topic) & (df['group'] == group)]
                score_lower = df_tmp[f'diff_{metric}_lower'].item()
                score_upper = df_tmp[f'diff_{metric}_upper'].item()
                sign = '' if score_lower < 0 < score_upper else '*'
                row.append(sign)
                # row.append(sign + '\n' + '[' + str(round(score_lower, 3)) + ', ' + str(round(score_upper, 3)) + ']')
            results.append(row)

        ci_matrix = pd.DataFrame(results)
        ci_matrix.columns = group_abbrev
        ci_matrix.index = topics_name
        data[metric] = ci_matrix
    return data


def pool_heatmaps():
    df = run('experiment_all_folds_no_selection', 'Gemma3_12B')
    df['var_pool'] = df['var_pool'].apply(lambda s: '+'.join(w[0].upper() for w in s.split('+')))
    df['topic'] = df['topic'].apply(lambda s: ' '.join(w.capitalize() for w in s.split('_')))

    model_prefix = {'llm': '', 'rf': 'rf_'}
    heatmap_data = {}
    signif_map = head_head_signif()

    def plot_heatmap_data(heat_matrix, heat_matrix_annot, fmt, center, cmap, filename):
        plt.figure(figsize=(10, 7))
        ax = sns.heatmap(
            heat_matrix,   
            center=center,
            annot=heat_matrix_annot,
            fmt=fmt,
            cmap=cmap,
            annot_kws={'size': 10},
            linecolor='white',
            linewidths=0.5,
            cbar=False
        )

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', labelrotation=0)
        plt.xlabel(None)
        plt.ylabel(None)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        #plt.title(filename)
        plt.show()


    for metric in ['f1_score', 'js', 'cramer_v']:
        for model, prefix in model_prefix.items():
            metric_name = prefix + metric
            heat_matrix = df.pivot(index='topic', columns='var_pool', values=metric_name)
            heatmap_data[metric_name] = heat_matrix
            # plot_heatmap_data(heat_matrix, heat_matrix, ".2f", 0.5, 'viridis', f'tex_src/images/heatmap_pool_metrics_{metric_name}.png')


    # diffs
    for metric in ['f1_score', 'js', 'cramer_v']:
        llm = heatmap_data[metric]
        rf = heatmap_data['rf_' + metric]

        if metric == 'js':
            diff_matrix = rf - llm
        else:
            diff_matrix = llm - rf

        diff_matrix_annot = diff_matrix.round(2).astype(str) + signif_map[metric]
        plot_heatmap_data(diff_matrix, diff_matrix_annot, '', 0, 'RdBu', f'tex_src/images/heatmap_pool_metrics_diff_{metric}.png')


def compare_select_no_select():
    df_all = compare_main_baseline('experiment_all_folds', 'Gemma3_12B')
    df_selec = compare_main_baseline('experiment_all_folds_no_selection', 'Gemma3_12B')
    results = []

    # print(df_all['JS'].mean())
    # print(df_selec['JS'].mean())

    for i in range(len(df_all)):
        f1_all = df_all.iloc[i]['F1-score']
        f1_selec = df_selec.iloc[i]['F1-score']
        f1_pool = df_all.iloc[i]['F1-score pool'] if f1_all > f1_selec else df_selec.iloc[i]['F1-score pool']# + ' [fs]'

        js_all = df_all.iloc[i]['JS']
        js_selec = df_selec.iloc[i]['JS']
        js_pool = df_all.iloc[i]['JS pool'] if js_all < js_selec else df_selec.iloc[i]['JS pool']# + ' [fs]'

        cramer_all = df_all.iloc[i]['Cramér’s V']
        cramer_selec = df_selec.iloc[i]['Cramér’s V']
        cramer_pool = df_all.iloc[i]['Cramér’s V pool'] if cramer_all > cramer_selec else df_selec.iloc[i]['Cramér’s V pool']# + ' [fs]'

        res = {
            'Topic': df_all.iloc[i]['Topic'],
            'Diff F1-score': f1_selec - f1_all,
            'Diff JS': js_all - js_selec,
            'Diff Cramér’s V': cramer_selec - cramer_all,
            'F1-score pool': f1_pool,
            'JS pool': js_pool,
            'Cramér’s pool': cramer_pool
        }
        results.append(res)

    df_res = pd.DataFrame(results)
    formatter_cells = {k: formatter_cell for k in df_res.keys() if 'Diff' in k}
    print(df_res.to_latex(index=False, escape=True, column_format='lcccccc', float_format="%.2f", formatters=formatter_cells))
    return df_res


if __name__ == '__main__':
    # topic = 'climate_change'
    # df = run('experiment_all_folds', 'Gemma3_12B')
    # print(','.join(df.columns))
    # df = df[df['topic'] == topic]
    # df = df.drop(['topic'], axis=1)
    # print(generate_latex_table(df, topic + 'llm_rf'))

    # df.set_index('var_pool', inplace=True)
    # heatmap(df, topic, 'heat.png')
    #var_pool_avg()
    #compare_main_baseline(folder='experiment_all_folds', model_prefix='Gemma3_12B')
    #compare_rf_llm_pool('experiment_all_folds')
    #pool_heatmaps()
    #compare_select_no_select()

    pool_heatmaps()



    

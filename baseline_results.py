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


def _get_metrics(files, col):
    js = 0
    f1_score = 0
    cramer_v = 0
    min_p_value = float('+inf')
    max_p_value = float('-inf')

    for file in files:
        df = pd.read_csv(file)
        actual, pred = get_probability_vectors(df, 'actual_answers', col)
        js+= jensenshannon(actual, pred, base=2)
        f1_score += f1_score_fn(df['actual_answers'], df[col], average='weighted')
        ct = pd.crosstab(df['actual_answers'], df[col])        
        _, p_value, _, _ = chi2_contingency(ct)

        if not col == 'const':
            cramer_v+= association(ct.to_numpy(), method="cramer", correction=True)

        if p_value < min_p_value:
            min_p_value = p_value
        if p_value > max_p_value:
            max_p_value = p_value

    n =  len(files)
    js = js/n
    f1_score = f1_score/n
    cramer_v = cramer_v/n
    return dict(js=js, f1_score=f1_score, cramer_v=cramer_v, min_p_value=min_p_value, max_p_value=max_p_value)


def run(folder, model_prefix):
    topics = get_topics()
    results = []

    for topic in topics:
        # random model
        files_random = glob(os.path.join(folder, f'random_{topic}_*.csv'))
        random_model_metrics = _get_metrics(files_random, 'random')

        files_const = glob(os.path.join(folder, f'const_{topic}_*.csv'))
        const_model_metrics = _get_metrics(files_const, 'const')

        files_llm = glob(os.path.join(folder, f'{model_prefix}_{topic}_*.csv'))
        llm_model_metrics = _get_metrics(files_llm, 'llm_answers')
        row = {'topic': topic}

        for metric, val in random_model_metrics.items():
            row['random_' + metric] = val

        for metric, val in const_model_metrics.items():
            # if metric not in ['min_p_value', 'max_p_value']:                
            row['const_' + metric] = val

        for metric, val in llm_model_metrics.items():
            row['llm_' + metric] = val
    
        results.append(row)
    return pd.DataFrame(results)


import pandas as pd

# Assuming you have already loaded your DataFrame as df
# Here's the function to generate the LaTeX table string:

def generate_latex_table(df):
    table = []
    table.append(r"\begin{table}[ht]")
    table.append(r"\centering")
    table.append(r"\caption{Performance comparison between LLM and baseline models. Reported metrics are averaged across folds. Lower Jensen–Shannon (JS) distance indicates better distributional alignment; higher Cramér’s V and F1-score indicate stronger predictive and associative performance.}")
    table.append(r"\begin{tabular}{llccc}")
    table.append(r"\toprule")
    table.append(r"\textbf{Topic} & \textbf{Model} & \textbf{F1-score} & \textbf{JS Distance} & \textbf{Cramér’s V} \\")
    table.append(r"\midrule")

    for topic in df['topic']:
        sub_df = df[df['topic'] == topic]
        table.append(f"\\textit{{{topic.replace('_', ' ').title()}}}     & LLM      & {sub_df['llm_f1_score'].values[0]:.2f}  & {sub_df['llm_js'].values[0]:.2f} & {sub_df['llm_cramer_v'].values[0]:.2f} \\\\")
        table.append(f"                   & Random   & {sub_df['random_f1_score'].values[0]:.2f}  & {sub_df['random_js'].values[0]:.2f} & {sub_df['random_cramer_v'].values[0]:.2f} \\\\")
        table.append(f"                   & Constant & {sub_df['const_f1_score'].values[0]:.2f}  & {sub_df['const_js'].values[0]:.2f} & {sub_df['const_cramer_v'].values[0]:.2f} \\\\")
        table.append(r"\addlinespace")

    table.append(r"\bottomrule")
    table.append(r"\end{tabular}")
    table.append(r"\label{tab:bench}")
    table.append(r"\end{table}")

    return '\n'.join(table)


def generate_latex_table_pvalue(df):
    def format_p(p):
        if p < 1e-10:
            return r"\textbf{< 1e-10}"
        elif p < 0.05:
            return f"\\textbf{{{p:.2e}}}"
        else:
            return f"{p:.2e}"

    latex = r"""\begin{table}[ht]
    \centering
    \caption{Minimum and maximum p-values from chi-squared independence tests across three validation folds, per topic and model. P-values below 0.05 are bolded. Lower values suggest stronger evidence against the null hypothesis of independence between predicted and actual responses.}
    \begin{tabular}{llcc}
    \toprule
    \textbf{Topic} & \textbf{Model} & \textbf{Min p-value} & \textbf{Max p-value} \\
    \midrule
    """

    for _, row in df.iterrows():
        topic = row['topic'].replace('_', ' ').title()
        latex += f"{topic} & LLM & {format_p(row['llm_min_p_value'])} & {format_p(row['llm_max_p_value'])} \\\\\n"
        latex += f"       & Random & {format_p(row['random_min_p_value'])} & {format_p(row['random_max_p_value'])} \\\\\n"
        latex += f"       & Constant & {format_p(row.get('const_min_p_value', 0))} & {format_p(row.get('const_max_p_value', 0))} \\\\\n"
        latex += r"\addlinespace" + "\n"

    latex += r"""\bottomrule
    \end{tabular}
    \label{tab:pvalue_comparison}
    \end{table}
    """
    return latex


if __name__ == '__main__':
    df = run('experiment_baseline', 'Gemma3_12B')
    print(df)
    # print(generate_latex_table(df))
    print(generate_latex_table_pvalue(df))
   
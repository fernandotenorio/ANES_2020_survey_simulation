import pandas as pd
import os
import re
from collections import defaultdict
from datagen import get_topics
from glob import glob
from utils import get_probability_vectors
from scipy.spatial.distance import jensenshannon
from scipy.stats.contingency import association
folder_path = 'experiment_free_vars'
topic_groups = defaultdict(list)
topics = sorted(get_topics())

for topic in topics:
    files = glob(os.path.join(folder_path, f'*{topic}*.csv'))
    for file in files:
        topic_groups[topic].append(file)

# Step 2: Voting per topic
results = []
results_js = {}
results_cramer = {}

for topic, files in topic_groups.items():
    llm_answers_list = []
    actual_answers = None

    for file in files:
        df = pd.read_csv(file)
        if actual_answers is None:
            actual_answers = df['actual_answers']
        llm_answers_list.append(df['llm_answers'])

    # Combine all LLM predictions into one DataFrame
    llm_df = pd.concat(llm_answers_list, axis=1)

    # Row-wise majority vote
    voted_answers = llm_df.mode(axis=1)[0]

    # Evaluate
    accuracy = (voted_answers == actual_answers).mean()

    # Js
    tmp_df = pd.DataFrame(dict(actual_answers=actual_answers, model_answers=voted_answers))
    actual, pred = get_probability_vectors(tmp_df, 'actual_answers', 'model_answers')
    js = jensenshannon(actual, pred, base=2)

    # Cramer V
    contingency_table = pd.crosstab(actual_answers, voted_answers)
    cramer_v = association(contingency_table.to_numpy(), method="cramer", correction=False)

    results.append(dict(topic=topic, acc=accuracy, cramer_v=cramer_v, js=js))


print(pd.DataFrame(results))

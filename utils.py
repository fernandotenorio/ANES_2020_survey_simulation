import re
import os
import pickle

from datagen import map_variables


def save_pickle(data, output_folder, filename):
    with open(os.path.join(output_folder, filename), 'wb') as f:
        pickle.dump(data, f)


def load_pickle(output_folder, filename):
    with open(os.path.join(output_folder, filename), 'rb') as f:
        data = pickle.load(f)
    return data


def deident_string(s):
    s = '\n'.join([m.lstrip() for m in s.split('\n')])
    return s.strip()


def aggregate_answer_for_climate_or_economy(topic, ans_code):
    topic_answer = None
    if topic == 'climate_change':
        if ans_code in [1]:
            topic_answer = 1
        elif ans_code in [2, 3]:
            topic_answer = 2
        elif ans_code in [4, 5]:
            topic_answer = 3
    elif topic == 'current_economy':
        if ans_code in [1, 2]:
            topic_answer = 1
        elif ans_code in [3]:
            topic_answer = 2
        elif ans_code in [4, 5]:
            topic_answer = 3
    else:
        topic_answer = ans_code
    return topic_answer


def generate_question_answer_template(topic):
    _, field_desc_map = map_variables()
    if topic not in field_desc_map:
        raise Exception(f"Topic {topic} not found.")

    if "question_format" not in field_desc_map[topic]:
        raise Exception(f"Topic {topic} missing question_format or question_answer fields.")

    question = field_desc_map[topic]["question_format"]
    choices = field_desc_map[topic]["question_answer"]

    s = 'Question:\n'
    s+= question + '\n\nAnswer choices:\n'

    for option, description in choices.items():
        s+= f'({str(option)}) {description}\n'

    pronoum = 'Your'
    s+=f'\n{pronoum} answer is option'
    return s.strip()


def extract_llm_answer(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else None


def get_probability_vectors(df, observed_col, predicted_col, possible_values=[1, 2, 3]):
    observed_counts = df[observed_col].value_counts(normalize=True)
    predicted_counts = df[predicted_col].value_counts(normalize=True)
    observed_probs = [observed_counts.get(value, 0.0) for value in possible_values]
    predicted_probs = [predicted_counts.get(value, 0.0) for value in possible_values]
    return observed_probs, predicted_probs
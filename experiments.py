from datagen import get_data, map_variables, get_topics, get_backstory_variables, split_data, k_fold_split_data
from ollama import chat
import re
import os
import random
import pandas as pd
import pickle
from pathlib import Path
import time
import numpy as np
from collections import Counter

from utils import save_pickle, load_pickle, deident_string, generate_question_answer_template, extract_llm_answer, aggregate_answer_for_climate_or_economy
from feature_selection import run_feature_selection, get_demographic_features, get_attitudinal_features, get_moral_features

from openai_client import ask_openai
from sklearn.ensemble import RandomForestClassifier


def format_persona(persona_dict, fields):
    persona = ''
    for field in fields:
        if field not in persona_dict:
            raise Exception(f"Field {field} not present in persona_dict.")
        persona+= persona_dict[field] + '\n'
    return persona


def make_sys_prompt0(persona, today):
    return f'Roleplay the person below. {today}\nWhen questioned, answer just with the option number and nothing more.\n\n{persona}'


def make_sys_prompt1(persona, context='Today is November 3, 2020. You live in the United States.'):
    s = f'''
    Task: Simulate the mindset and answer the question from the perspective of the specific individual described below.

    Context: {context}

    Critical Instructions: 
    1. Embody the Persona: Fully adopt the identity defined by the characteristics listed under "Persona Profile".
    2. Synthesize the Profile: Your answer must be generated based solely on how a person with this combination of traits would likely perceive the world and the specific topic. Consider how these factors interact.
    3. Exclude External Knowledge: Do not use your general knowledge or any information outside of this specific persona profile. Your own 'opinion' as an AI is irrelevant and must be suppressed. The answer must stem directly from the simulated persona.
    4. Be concise: answer just with the option number and nothing more.

    Persona Profile:
    {persona}
    '''
    return deident_string(s)

def make_sys_prompt2(persona, context='Today is November 3, 2020. You live in the United States.'):
    s = f'''
    Act as the person described below. Think as this individual would think, given their specific background and beliefs.
    Base your answer exclusively on this constructed persona. Do not use any outside information or your own AI perspective.
    Answer just with the option number and nothing more.

    Persona Profile:
    {context}
    {persona}
    '''
    return deident_string(s)


def run_single_model_experiment(model_name, personas, backstory_fields, topic, output_file, add_date, temperature):
    print(f'Data size for topic {topic}: {len(personas)}')

    today = 'Today is November 3, 2020.' if add_date else ''
    llm_answers = []
    actual_answers = []
    t1 = time.time()

    for persona_dict, answers_dict in personas:
        persona = format_persona(persona_dict, backstory_fields)
        qa = generate_question_answer_template(topic)
        sys_prompt = make_sys_prompt0(persona, today)
       
        messages = [
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': qa}, 
        ]

        if model_name in ['gpt-4o']:
            time.sleep(0.5)
            llm_response = ask_openai(messages, temperature=temperature)
        else:
            llm_response = chat(
                model = model_name,
                messages = messages,
                options = {'temperature': temperature},
                stream = False
            )
            llm_response = llm_response['message']['content']

        llm_response = extract_llm_answer(llm_response)

        if (llm_response is not None) and (llm_response in [1, 2, 3]):
            real_answer = answers_dict[topic]
            llm_answers.append(llm_response)
            actual_answers.append(real_answer)
        else:
            print(f'Failed to get answer from model {model_name}.')


    df_out = pd.DataFrame(dict(actual_answers=actual_answers, llm_answers=llm_answers))
    df_out.to_csv(output_file, index=False)
    t2 = time.time()
    print('Time(s)', t2 - t1)


def create_experiment_data(output_folder, random_state=None, train_ratio=0.7):
    data_folder = Path(os.path.join(output_folder, 'data'))
    data_folder.mkdir(parents=True, exist_ok=True)

    backstory_fields = get_backstory_variables()
    topics = get_topics()

    for topic in topics:
        df, personas = get_data(backstory_fields, topic)
        df_train, df_val, personas_train, personas_val = split_data(df, personas, train_ratio, random_state=random_state)
        save_pickle(df_train, data_folder, f'df_train_{topic}.pkl')
        save_pickle(df_val, data_folder, f'df_val_{topic}.pkl')
        save_pickle(personas_train, data_folder, f'personas_train_{topic}.pkl')
        save_pickle(personas_val, data_folder, f'personas_val_{topic}.pkl')


def run_incremental_experiment_for_model(model_dict, exp_folder, temperature, single):
    ''' all variables available'''
    topics = get_topics()
    model_name = model_dict['model_name']
    model_alias = model_dict['model_alias']

    variable_groups = [
                ('demo', dict(demographic=True, attitudinal=False, moral=False)),
                ('demo+attit', dict(demographic=True, attitudinal=True, moral=False)),
                ('demo+attit+moral', dict(demographic=True, attitudinal=True, moral=True)),
                ('attit', dict(demographic=False, attitudinal=True, moral=False)),
                ('attit+moral', dict(demographic=False, attitudinal=True, moral=True)),
                ('demo+moral', dict(demographic=True, attitudinal=False, moral=True)),
                ('moral', dict(demographic=False, attitudinal=False, moral=True))
    ]

    for group_label, group_dict in variable_groups:
        for topic in topics:
            data_folder = os.path.join(exp_folder, 'data')
            # LLM runs on validation set
            personas_val = load_pickle(data_folder, f'personas_val_{topic}.pkl')
            output_file = os.path.join(exp_folder, f'{model_alias}_{topic}_T{temperature}_{group_label}.csv')
            selected_backstory_fields = run_feature_selection(model_name, topic, temperature, **group_dict, single=single)
            print(f'Selected backstory fields for topic {topic}: {selected_backstory_fields}')
            run_single_model_experiment(model_name, personas_val, selected_backstory_fields, topic, output_file, add_date=True, temperature=temperature)


def run_single_experiment_for_model(model_dict, exp_folder, temperature):
    ''' all variables available'''
    topics = get_topics()
    model_name = model_dict['model_name']
    model_alias = model_dict['model_alias']
   
    for topic in topics:
        data_folder = os.path.join(exp_folder, 'data')
        # LLM runs on validation set
        personas_val = load_pickle(data_folder, f'personas_val_{topic}.pkl')
        output_file = os.path.join(exp_folder, f'{model_alias}_{topic}_T{temperature}.csv')
        selected_backstory_fields = run_feature_selection(model_name, topic, temperature, single=True)
        print(f'Selected backstory fields for topic {topic}: {selected_backstory_fields}')
        run_single_model_experiment(model_name, personas_val, selected_backstory_fields, topic, output_file, add_date=True, temperature=temperature)


def run_baseline_experiment(model_dict, exp_folder, temperature, random_state):
    data_folder = Path(os.path.join(exp_folder, 'data'))
    data_folder.mkdir(parents=True, exist_ok=True)

    backstory_fields = get_backstory_variables()
    topics = get_topics()
    model_name = model_dict['model_name']
    model_alias = model_dict['model_alias']
    k = 3

    for topic in topics:
        df, personas = get_data(backstory_fields, topic)
        folds = k_fold_split_data(df, personas, n_folds=k, random_state=random_state, shuffle=True)

        for i, (df_train, df_val, personas_train, personas_val) in enumerate(folds):
            save_pickle(df_train, data_folder, f'df_train_{topic}_{i}.pkl')
            save_pickle(df_val, data_folder, f'df_val_{topic}_{i}.pkl')
            save_pickle(personas_train, data_folder, f'personas_train_{topic}_{i}.pkl')
            save_pickle(personas_val, data_folder, f'personas_val_{topic}_{i}.pkl')

    # run models
    for topic in topics:
        selected_backstory_fields = run_feature_selection(model_name, topic, temperature, single=True)
        print(f'Selected backstory fields for topic {topic}: {selected_backstory_fields}')

        for i in range(k):
            personas_val = load_pickle(data_folder, f'personas_val_{topic}_{i}.pkl')
            output_file = os.path.join(exp_folder, f'{model_alias}_{topic}_T{temperature}_{i}.csv')
            run_single_model_experiment(model_name, personas_val, selected_backstory_fields, topic, output_file, add_date=True, temperature=temperature)

            # train data
            personas_train = load_pickle(data_folder, f'personas_train_{topic}_{i}.pkl')
            train_answers = [a[topic] for _, a in personas_train]
            train_answers = Counter(train_answers)
            most_freq = train_answers.most_common(1)[0][0]
            real_answers = [a[topic] for _, a in personas_val]

            run_base = False

            if run_base:
                # const model
                df_const = pd.DataFrame()
                df_const['actual_answers'] = real_answers
                df_const['const'] = most_freq
                const_output_file = os.path.join(exp_folder, f'const_{topic}_{i}.csv')
                df_const.to_csv(const_output_file, index=False)

                # random model
                df_random = pd.DataFrame()
                df_random['actual_answers'] = real_answers
                df_random['random'] = np.random.choice([1, 2, 3], size=len(real_answers))
                random_output_file = os.path.join(exp_folder, f'random_{topic}_{i}.csv')
                df_random.to_csv(random_output_file, index=False)


def run_all_folds_experiment(model_dict, exp_folder, temperature, random_state):
    data_folder = Path(os.path.join(exp_folder, 'data'))
    data_folder.mkdir(parents=True, exist_ok=True)

    backstory_fields = get_backstory_variables()
    topics = get_topics()
    model_name = model_dict['model_name']
    model_alias = model_dict['model_alias']
    k = 3

    # For RF model
    demo_feats = get_demographic_features()
    attit_feats = get_attitudinal_features()
    moral_feats = get_moral_features()

    for topic in topics:
        df, personas = get_data(backstory_fields, topic)      
        folds = k_fold_split_data(df, personas, n_folds=k, random_state=random_state, shuffle=True)

        for i, (df_train, df_val, personas_train, personas_val) in enumerate(folds):
            save_pickle(df_train, data_folder, f'df_train_{topic}_{i}.pkl')
            save_pickle(df_val, data_folder, f'df_val_{topic}_{i}.pkl')
            save_pickle(personas_train, data_folder, f'personas_train_{topic}_{i}.pkl')
            save_pickle(personas_val, data_folder, f'personas_val_{topic}_{i}.pkl')

    variable_groups = [
                ('demo', dict(demographic=True, attitudinal=False, moral=False), demo_feats),
                ('demo+attit', dict(demographic=True, attitudinal=True, moral=False), demo_feats + attit_feats),
                ('demo+attit+moral', dict(demographic=True, attitudinal=True, moral=True), demo_feats + attit_feats + moral_feats),
                ('attit', dict(demographic=False, attitudinal=True, moral=False), attit_feats),
                ('attit+moral', dict(demographic=False, attitudinal=True, moral=True), attit_feats + moral_feats),
                ('demo+moral', dict(demographic=True, attitudinal=False, moral=True), demo_feats + moral_feats),
                ('moral', dict(demographic=False, attitudinal=False, moral=True), moral_feats)
    ]
    # run models
    for group_label, group_dict, rf_feats in variable_groups:
        for topic in topics:            
            #selected_backstory_fields = run_feature_selection(model_name, topic, temperature, **group_dict, single=False)
            selected_backstory_fields = rf_feats
            print('Note: LLM using all features from pool!')
            print(f'Selected backstory fields for topic {topic}: {selected_backstory_fields}')

            for i in range(k):
                personas_val = load_pickle(data_folder, f'personas_val_{topic}_{i}.pkl')
                output_file = os.path.join(exp_folder, f'{model_alias}_{topic}_T{temperature}_{i}_{group_label}.csv')
                run_single_model_experiment(model_name, personas_val, selected_backstory_fields, topic, output_file, add_date=True, temperature=temperature)

                train_rf = False

                if train_rf:
                    df_train = load_pickle(data_folder, f'df_train_{topic}_{i}.pkl')
                    df_val = load_pickle(data_folder, f'df_val_{topic}_{i}.pkl')

                    X_train = df_train[rf_feats]
                    X_val = df_val[rf_feats]
                    y_train = df_train[topic]
                    y_val = df_val[topic]

                    # aggregate answers for climate_change and current_economy, just like for the LLM
                    if topic in ['climate_change', 'current_economy']:
                        for j in range(len(y_train)):
                            y_train[j] = aggregate_answer_for_climate_or_economy(topic, y_train[j])
                      
                        for j in range(len(y_val)):
                            y_val[j] = aggregate_answer_for_climate_or_economy(topic, y_val[j])

                    model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=2, random_state=random_state)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)

                    df_rf = pd.DataFrame()
                    df_rf['actual_answers'] = y_val
                    df_rf['RF_answers'] = y_pred
                    rf_output_file = os.path.join(exp_folder, f'RF_{topic}_{i}_{group_label}.csv')
                    df_rf.to_csv(rf_output_file, index=False)
            
                

if __name__ == '__main__':
    model_dict = {'model_name': 'gemma3:12b-it-q4_K_M', 'model_alias': 'Gemma3_12B'}
    exp_folder = 'experiment_all_folds'
    temperature = 0.3
    #run_baseline_experiment(model_dict, exp_folder, temperature, 42)
    run_all_folds_experiment(model_dict, exp_folder, temperature, 42)



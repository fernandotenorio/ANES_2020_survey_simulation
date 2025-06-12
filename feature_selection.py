from datagen import get_topics, get_backstory_variables
from ollama import chat
import re
import os
import random
from pydantic import BaseModel
import pandas as pd
import time

from utils import deident_string, generate_question_answer_template
from openai_client import ask_openai



class FeatureSelectionModel(BaseModel):
    selected_variables: list[str]

def topic_nice(topic):
    m = {
        'race_diversity': 'Race diversity',
        'gender_role': 'Gender role',
        'current_economy': 'Current Economy',
        'drug_addiction': 'Drug addiction',
        'climate_change': 'Climate change',
        'gay_marriage': 'Gay marriage',
        'refugee_allowing': 'Refugee policy',
        'health_insurance_policy': 'Health insurance policy',
        'gun_regulation': 'Gun regulation',
        'income_inequality': 'Income inequality'
    }
    return m[topic]


def make_specialist_sys_prompt():
    return 'You are a human behavior and psychology expert.'

def get_demographic_features():
    return ['race', 'age', 'gender', 'income', 'education', 'occupation', 'city_rural', 'children', 'health_insurance']

def get_attitudinal_features():
    return ['ideology', 'party', 'interested_politics', 'trust_media', 'science', 'vaccines_autism', 'religion_importance']

def get_moral_features():
    return ['child_trait', 'death_penalty', 'birth_citizenship', 'children_sent_back', 'discrimination_woman', 'black_discrimination', 'black_hist', 'gays_discrimination', 'muslins_discrimination']


def make_feature_selection_prompt(topic, question, demographic=True, attitudinal=True, moral=True, single=False):
    assert (demographic or attitudinal or moral)
    s = f'''
    You are a reasoning assistant trained to understand how personal characteristics shape beliefs and opinions.

    Your task is to analyze a set of variables describing individuals in the U.S. and select the variables most predictive of a person's response to a survey question on the topic of **"{topic}"**.

    Below is the list of variables, grouped by theme, along with their descriptions:
    '''

    if demographic:
        s+= f'''
        ### Demographic
        - race: Self-identified race
        - age: Age
        - gender: Male or female
        - income: Annual family income
        - education: Education level
        - occupation: Employment type
        - city_rural: Identifies as urban or rural
        - children: Number of children
        - health_insurance: Has health insurance or not
        '''

    if attitudinal:
        s+= f'''
        ### Attitudes & Political Orientation
        - ideology: Political ideology (liberal, conservative, etc.)
        - party: Political party affiliation
        - interested_politics: Interested in politics or not
        - trust_media: Trust in mainstream media
        - science: Belief that people need help from experts to understand science
        - vaccines_autism: Belief that vaccines cause autism
        - religion_importance: How important religion is in the respondent’s life
        '''

    if moral:
        s+= f'''
        ### Moral Compass & Social Values
        - child_trait: Most important child trait (e.g., obedience vs. self-reliance)
        - death_penalty: Support or opposition to the death penalty
        - birth_citizenship: View on banning birthright citizenship for children of undocumented immigrants
        - children_sent_back: View on deporting children of undocumented immigrants
        - discrimination_woman: Perceived discrimination against women
        - black_discrimination: Perceived discrimination against Black people
        - black_hist: Belief that past racism still affects Black Americans today
        - gays_discrimination: Perceived discrimination against gay people
        - muslins_discrimination: Perceived discrimination against Muslims
        '''

    s+= f'''
    ### Survey Question
    We want to predict the respondent’s answer to the following question:
    > {question}
    '''

    if single:
        s+= f'''
        ### Instructions
        Based on your understanding of human behavior, culture, and political psychology, select **exactly one variable from each of the three groups** above that you believe is the most predictive of how someone would respond to the question above.
        Your goal is to identify the **single most informative variable** from each group — no more, no less — to support a high-quality prediction with minimal input.

        Respond only with your final answer in **this JSON format**:

        {{
          "selected_variables": ["variable_1", "variable_2", "variable_3"]
        }}

        Do not include any explanation or commentary.
        '''
    else:
        s+= f'''
        ### Instructions
        Based on your understanding of human behavior, culture, and political psychology, select **only the variables that are most predictive** of how someone would respond to the question above.

        You may select **as few or as many variables as you believe are truly necessary** to make a high-quality prediction — but favor **clarity and precision** over length.

        Respond only with your final answer in **this JSON format**:

        {{
          "selected_variables": ["variable_1", "variable_2", "..."]
        }}

        Do not include any explanation or commentary.
        '''
    return deident_string(s)


def run_feature_selection(model, topic, temperature, demographic=True, attitudinal=True, moral=True, single=False, max_tries=10):
    backstory_variables = get_backstory_variables()
    for _ in range(max_tries):
        try:
            qa = generate_question_answer_template(topic).replace('Your answer is option', '').strip()
            fs_prompt = make_feature_selection_prompt(topic_nice(topic), qa, demographic, attitudinal, moral, single)            
            messages = [
                        {'role': 'user', 'content': fs_prompt}, 
            ]

            if model in ['gpt-4o']:
                time.sleep(0.5)
                llm_response = ask_openai(messages, temperature=temperature)
            else:
                llm_response = chat(
                    model = model,
                    messages = messages,
                    format = FeatureSelectionModel.model_json_schema(),
                    options = {'temperature': temperature},
                    stream = False
                )
                llm_response = llm_response['message']['content']                

            answer = FeatureSelectionModel.model_validate_json(llm_response)            
            selected_variables = answer.selected_variables

            if selected_variables and all(v in backstory_variables for v in selected_variables):
                return selected_variables
        except Exception as e:
            print(f'Feature selection failed for topic {topic}')
            continue

assert set(get_backstory_variables()) == set(get_demographic_features() + get_attitudinal_features() + get_moral_features()),  'Features mismatch'

if __name__ == '__main__':
    pass
import pandas as pd
import random
import numpy as np
import os
from sklearn.model_selection import KFold

def map_variables(first_person=False):
    fp_self = "I"
    fp_tobe = "am"
    fp_pos_self = "my"
    fp_reflex_self = "myself"

    tp_self = "you"
    tp_tobe = "are"
    tp_pos_self = "your"
    tp_reflex_self = "yourself"

    if first_person:
        p_self = fp_self
        p_tobe = fp_tobe
        p_pos_self = fp_pos_self
        p_reflex_self = fp_reflex_self
    else:
        p_self = tp_self
        p_tobe = tp_tobe
        p_pos_self = tp_pos_self
        p_reflex_self = tp_reflex_self


    fields_map = {

        'V200001' : {
            'field_desc': 'CASE_ID',
            'field_code': 'V200001'
        },

        'V201549x': {   #-9. Refused, -8. Don’t know
            "template": f"Racially, {p_self} {p_tobe} XXX.",
            "valmap": {1: 'White', 2: 'Black', 3: 'Hispanic', 4: 'Asian', 5: 'Native American', 6: 'Mixed'},
            "field_desc" : "race",
            "field_code" : "V201549x"
        },

        # 'V202022': {
        #     "template": "XXX.",
        #     "valmap": {
        #         1: f'{p_self.capitalize()} like to discuss politics with {p_pos_self} family and friends',
        #         2: f'{p_self.capitalize()} never discuss politics with {p_pos_self} family or friends'
        #     },
        #     "field_desc" : "discuss_politics",
        #     "field_code" : "V202022"
        # },

        'V201200': {
            "template": f"Ideologically, XXX.",
            "valmap": {
                1: "You are extremely liberal",
                2: "You are liberal",
                3: "You are slightly liberal",
                4: "You are moderate",
                5: "You are slightly conservative",
                6: "You are conservative",
                7: "You are extremely conservative",
                99: "You haven't thought much about it"
            },
            "field_desc" : "ideology",
            "field_code" : "V201200"
        },

        'V201231x': {
            "template": f"Politically, {p_self} {p_tobe} XXX.",
            "valmap": {
                1: "a strong Democrat",
                2: "not very strong Democrat",
                3: "an independent who leans Democratic",
                4: "an independent",
                5: "an independent who leans Republican",
                6: "not very strong Republican",
                7: "a strong Republican"
            },
            "field_desc" : "party",
            "field_code" : "V201231x"
        },
        # 'V201452': {
        #     "template": f"{p_self.capitalize()} XXX.",
        #     "valmap": {1: "attend church", 2: "do not attend church"},
        #     "field_desc" : "church",
        #     "field_code" : "V201452"
        # },
        # 'V201453': {
        #     "template": f"{p_self.capitalize()} XXX.",
        #     "valmap": {1: "attend religious services every week", 2: "attend religious services almost every week", 3: "attend religious services once or twice a month", 4: "attend religious services a few times a year", 5: "never attend religious services"},
        #     "field_desc" : "religious_services",
        #     "field_code" : "V201453"
        # },

        'V201433': { #-9. Refused -8. Don’t know
            "template": f"Religion is XXX in my life.",
            "valmap": {1: "extremely important", 2: "very important", 3: "moderately important", 4: "of little importance", 5: "not important at all"},
            "field_desc" : "religion_importance",
            "field_code" : "V201433"
        },

        'V201507x': { #-9. Refused, 80. Age 80 or older
            "template": f"{p_self.capitalize()} {p_tobe} XXX years old.",
            "valmap": {},
            "field_desc" : "age",
            "field_code" : "V201507x"
        },
        'V201600': { #-9. Refused
            "template": f"{p_self.capitalize()} {p_tobe} a XXX.",
            "valmap": {1: "man", 2: "woman"},
            "field_desc" : "gender",
            "field_code" : "V201600"
        },
        'V202406': { ##-9. Refused -8. Don’t know, 
            "template": f"{p_self.capitalize()} {p_tobe} XXX in politics.",
            "valmap": {1: "very interested", 2: "somewhat interested", 3: "not very interested", 4: "not at all interested"},
            "field_desc" : "interested_politics",
            "field_code" : "V202406"
        },
        'V201617x':{ # TOTAL (FAMILY) INCOME, -9. Refused, -5. Interview breakoff
            "template": f"{p_pos_self.capitalize()} total family income is XXX.",
            "valmap": {1: "Under $9,999", 2: "$10,000-14,999", 3:"$15,000-19,999", 4:"$20,000-24,999", 5:"$25,000-29,999", 6:"$30,000-34,999",
                       7: "$35,000-39,999", 8: "$40,000-44,999", 9: "$45,000-49,999", 10:"$50,000-59,999", 11:"$60,000-64,999", 12:"$65,000-69,999",
                       13: "$70,000-74,999", 14:"$75,000-79,999", 15:"$80,000-89,999", 16:"$90,000-99,999", 17:"$100,000-109,999", 18:"$110,000-124,999",
                       19: "$125,000-149,999", 20:"$150,000-174,999", 21:"$175,000-249,999", 22:"$250,000 or more"
            },
            "field_desc" : "income",
            "field_code" : "V201617x"
        },
        'V201620': {
            "template": f"At the moment {p_self} XXX.",
            "valmap" : {1: "do have health insurance", 2: "do not have health insurance"},
            "field_desc" : "health_insurance",
            "field_code" : "V201620"
        },

        'V201511x' : {
            "template": f"{p_self.capitalize()} have XXX.",
            "valmap" : {1: "less than high school credential", 2: "a high school credential", 3: "some post-high school, no bachelor’s degree", 
                        4: "a Bachelor’s degree", 5: "a Graduate degree"
            },
            "field_desc" : "education",
            "field_code" : "V201511x"
        },
        'V201529' : {
            "template": f"{p_pos_self.capitalize()} employment is best described as working for XXX.",
            "valmap" : { 1: "a for-profit company or organization", 2: "a non-profit organization", 3: "the local government", 4: "the State government",
                        5: "the Military", 6: "the Federal government, as a civilian employee", 7: "yourself, as owner of non-incorporated business",
                        8: "yourself, as owner of incorporated business", 9: "a for-profit family business, without pay in"
            },
            "field_desc" : "occupation",
            "field_code" : "V201529"
        },

        'V202356' : {
             "template": f"{p_self.capitalize()} consider {p_reflex_self} XXX.",
             "valmap" : {1: "a city person", 2: "a suburb person", 3: "a small-town person", 4: "a country person", 5: "neither a city nor rural person"},
             "field_desc" : "city_rural",
             "field_code" : "V202356"
        },

        'V201567' : {
            "template": f"{p_self.capitalize()} have XXX.",
             "valmap" : {0: "no children", 1: "one child", 2: "two children", 3: "three children", 4: "four or more children" },
             "field_desc" : "children",
             "field_code" : "V201567"
        },

        'V202309' : {
            "template": f"{p_self.capitalize()} believe that people XXX from experts to understand science.",
             "valmap" : {1: "do not need help", 2: "need little or no help", 3: "need a moderate amount of help", 4: "need a lot of help", 5: "need all the help possible" },
             "field_desc" : "science",
             "field_code" : "V202309"
        },

        # 'V202333' : {
        #     "template": f"{p_self.capitalize()} believe that the issue of climate change is XXX.",
        #     "valmap" : {1: "not at all important", 2: "a little important", 3: "moderately important", 4: "very important", 5: "extremely important" },
        #     "field_desc" : "climate_change_importance",
        #     "field_code" : "V202333"
        # },

        'V202553' : {
            "template": f"{p_self.capitalize()} believe that XXX.",
            "valmap" : {1: "most scientific evidence shows childhood vaccines cause autism", 2: "most scientific evidence shows childhood vaccines do not cause autism" },
            "field_desc" : "vaccines_autism",
            "field_code" : "V202553"
        },

        'V201377' : {
            "template": f"{p_self.capitalize()} have XXX trust in the media when it comes to reporting the news accurately and fairly.",
            "valmap" : {1: "zero", 2: "little", 3: "moderate", 4: "a lot of", 5: "a great deal of"},
            "field_desc" : "trust_media",
            "field_code" : "V201377"
        },

        # 'V202264' : {
        #     "template": f"{p_self.capitalize()} XXX that the world is always changing and we should adjust our view of moral behavior to those changes.",
        #     "valmap" : {1: "agree strongly", 2: "agree somewhat", 3: "neither agree nor disagree", 4: "disagree somewhat", 5: "disagree strongly"},
        #     "field_desc" : "world_changing",
        #     "field_code" : "V202264"
        # },

        'V202268' : {
            "template": f"{p_self.capitalize()} believe that when it comes to obedience vs self-reliance, XXX is a more important trait for a child to have.",
            "valmap" : {1: "obedience", 2: "self-reliance", 3: "both", 4: "neither"},
            "field_desc" : "child_trait",
            "field_code" : "V202268"
        },

        'V202534' : {
            "template": f"{p_self.capitalize()} believe there is XXX discrimination against women in the US today.",
            "valmap" : {1: "a great deal of", 2: "a lot of", 3: "a moderate amount of", 4: "little", 5: "no"},
            "field_desc" : "discrimination_woman",
            "field_code" : "V202534"
        },

        'V201420x' : {
            "template": f"{p_self.capitalize()} XXX that children of unauthorized immigrants do not automatically get citizenship if they are born in the US.",
            "valmap" : {1: "favor a great deal", 2: "favor a moderate amount", 3: "favor a little", 4: "neither favor nor oppose", 5: "oppose a little", 6: "oppose a moderate amount", 7: "oppose a great deal"},
            "field_desc" : "birth_citizenship",
            "field_code" : "V201420x"
        },

        'V201423x' : {
            "template": f"{p_self.capitalize()} XXX that children of unauthorized immigrants who were brought to the US illegally and have lived here for at least 10 years should be deported.",
            "valmap" : {1: "favor a great deal", 2: "favor a moderate amount", 3: "favor a little", 4: "oppose a little", 5: "oppose a moderate amount", 6: "oppose a great deal"},
            "field_desc" : "children_sent_back",
            "field_code" : "V201423x"
        },

        'V202301' : {
            "template": f"{p_self.capitalize()} XXX that generations of slavery and discrimination have created conditions that make it difficult for blacks to work their way out of the lower class.",
            "valmap" : {1: "agree strongly", 2: "agree somewhat", 3: "neither agree nor disagree", 4: "disagree somewhat", 5: "disagree strongly"},
            "field_desc" : "black_hist",
            "field_code" : "V202301"
        },

        'V202527' : {
            "template": f"{p_self.capitalize()} believe there is XXX discrimination against blacks in the US today.",
            "valmap" : {1: "a great deal of", 2: "a lot of", 3: "a moderate amount of", 4: "little", 5: "no"},
            "field_desc" : "black_discrimination",
            "field_code" : "V202527"
        },

        'V202533' : {
            "template": f"{p_self.capitalize()} believe there is XXX discrimination against gays and lesbians in the US today.",
            "valmap" : {1: "a great deal of", 2: "a lot of", 3: "a moderate amount of", 4: "little", 5: "no"},
            "field_desc" : "gays_discrimination",
            "field_code" : "V202533"
        },

        'V202531' : {
            "template": f"{p_self.capitalize()} believe there is XXX discrimination against muslins in the US today.",
            "valmap" : {1: "a great deal of", 2: "a lot of", 3: "a moderate amount of", 4: "little", 5: "no"},
            "field_desc" : "muslins_discrimination",
            "field_code" : "V202531"
        },

        'V201343' : {
            "template": f"{p_self.capitalize()} XXX the death penalty.",
            "valmap" : {1: "favor", 2: "oppose"},
            "field_desc" : "death_penalty",
            "field_code" : "V201343"
        },

        # 'topic' variables
        'V202371' : {
            "template": f"{p_self.capitalize()} believe that increasing the number of people from different races and ethnic groups in the United States XXX.",
            "valmap" : {1: 'makes this country a better place to live', 2: 'makes this country a worse place to live', 3: 'makes no difference'},
            "question_format": "Does the increasing number of people of many different races and ethnic groups in the United States make this country a better place to live, a worse place to live, or does it make no difference?",
            "question_answer": {1: 'Better', 2: 'Worse', 3: 'Makes no difference'},
            "field_desc" : "race_diversity",
            "field_code" : "V202371"
        },

        'V202287' : {
            "template": f"{p_self.capitalize()} believe that it XXX for the family as a whole if the man works outside the home and the woman takes care of the home and family.",
            "valmap" : {1: 'is better', 2: 'is worse', 3: 'makes no difference'},
            "question_format": "Do you think it is better, worse, or makes no difference for the family as a whole if the man works outside the home and the woman takes care of the home and family?",
            "question_answer": {1: 'Better', 2: 'Worse', 3: 'Makes no difference'},
            "field_desc" : "gender_role",
            "field_code" : "V202287"
        },

        'V201324' : {
            "template": f"{p_self.capitalize()} think the state of the Economy these days in the United States is XXX.",
            "valmap" : {1: 'good', 2: 'neither good nor bad', 3: 'bad'},
            "question_format": "What do you think about the state of the economy these days in the United States?",
            "question_answer": {1: 'Good', 2: 'Neither good nor bad', 3: 'Bad'},
            "field_desc" : "current_economy",
            "field_code" : "V201324"
        },

        'V202348' : {
            "template": f"{p_self.capitalize()} think the federal government XXX about the opioid drug addiction issue.",
            "valmap" : {1: 'should be doing more', 2: 'should be doing less', 3: 'is doing the right amount'},
            "question_format": "Do you think the federal government should be doing more about the opioid drug addiction issue, should be doing less, or is it currently doing the right amount?",
            "question_answer": {1: 'Should be doing more', 2: 'Should be doing less', 3: 'Is doing the right amount'},
            "field_desc" : "drug_addiction",
            "field_code" : "V202348"
        },

        'V202332' : {
            "template": f"{p_self.capitalize()} think that climate change is XXX severe weather events or temperature patterns in the United States.",
            "valmap" : {1: 'not at all related to', 2: 'somewhat related to', 3: 'strongly related to'},
            "question_format": "How much, if at all, do you think climate change is currently affecting severe weather events or temperature patterns in the United States?",
            "question_answer": {1: 'Not at all', 2: 'A little', 3: 'A lot'},
            "field_desc" : "climate_change",
            "field_code" : "V202332"
        },

        'V201416' : {
            "template": f"{p_self.capitalize()} believe that gay and lesbian couples XXX.",
            "valmap" : {1: 'should be allowed to legally marry', 2: 'should be allowed to form civil unions but not legally marry', 3: 'should have no legal recognition of their relationship'},
            "question_format": "Which comes closest to your view? You can just tell me the number of your choice.",
            "question_answer": {1: 'Gay and lesbian couples should be allowed to legally marry.', 2: 'Gay and lesbian couples should be allowed to form civil unions but not legally marry', 3: 'There should be no legal recognition of gay or lesbian couples’ relationship.'},
            "field_desc" : "gay_marriage",
            "field_code" : "V201416"
        },

        'V202234' : {
            "template": f"{p_self.capitalize()} XXX allowing refugees who are fleeing war, persecution, or natural disasters in other countries to come to live in the United States.",
            "valmap" : {1: 'favor', 2: 'oppose', 3: 'neither favor nor oppose'},
            "question_format": "Do you favor, oppose, or neither favor nor oppose allowing refugees who are fleeing war, persecution, or natural disasters in other countries to come to live in the U.S.?",
            "question_answer": {1: 'Favor', 2: 'Oppose', 3: 'Neither favor nor oppose'},
            "field_desc" : "refugee_allowing",
            "field_code" : "V202234"
        },

        'V202378' : {
            "template": f"{p_self.capitalize()} XXX in government spending to help people pay for health insurance when people cannot pay for it all themselves.",
            "valmap" : {1: 'an increase', 2: 'a decrease', 3: 'no change'},
            "question_format": "Do you favor an increase, decrease, or no change in government spending to help people pay for health insurance when people cannot pay for it all themselves?",
            "question_answer": {1: 'Increase', 2: 'Decrease', 3: 'No change'},
            "field_desc" : "health_insurance_policy",
            "field_code" : "V202378"
        },

        'V202337' : {
            "template": f"{p_self.capitalize()} think the federal government should make the rules XXX for people to buy a gun.",
            "valmap" : {1: 'more difficult', 2: 'easier', 3: 'about the same as they are now'},
            "question_format": "Do you think the federal government should make it more difficult for people to buy a gun than it is now, make it easier for people to buy a gun, or keep these rules about the same as they are now?",
            "question_answer": {1: 'More difficult', 2: 'Easier', 3: 'Keep these rules about the same'},
            "field_desc" : "gun_regulation",
            "field_code" : "V202337"
        },

        'V202257' : {
            "template": f"{p_self.capitalize()} XXX the government trying to reduce the difference in incomes between the richest and poorest households.",
            "valmap" : {1: 'favor', 2: 'oppose', 3: 'neither favor nor oppose'},
            "question_format": "Do you favor, oppose, or neither favor nor oppose the government trying to reduce the difference in incomes between the richest and poorest households?",
            "question_answer": {1: 'Favor', 2: 'Oppose', 3: 'Neither favor nor oppose'},
            "field_desc" : "income_inequality",
            "field_code" : "V202257"
        }
    }

    # sanity check
    assert all(k == v['field_code'] for k, v in fields_map.items())
    field_desc_map = {v['field_desc'] : v for k, v in fields_map.items()}
    return fields_map, field_desc_map


def get_topics():
    fields_map, _ = map_variables()
    topics = [d["field_desc"] for d in fields_map.values() if 'question_answer' in d]
    return topics


def get_backstory_variables():
    fields_map, _ = map_variables()
    backstory_fields = [d["field_desc"] for d in fields_map.values() if 'template' in d and 'question_answer' not in d]
    return backstory_fields


def get_data(backstory_fields, topic):
    assert isinstance(backstory_fields, list)
    assert isinstance(topic, str)
    assert len(set(backstory_fields) & set([topic])) == 0

    data_folder = 'data'
    file = os.path.join(data_folder, 'anes_timeseries_2020_csv_20220210.csv')
    df = pd.read_csv(file, low_memory=False, encoding='utf-8-sig')

    # remove 2016-2020 Panel
    df = df[df['V200003'] != 2]
    fields_map, field_desc_map = map_variables()
    feature_cols = list(fields_map.keys())
    df = df[feature_cols]

    # drop CaseID
    df = df.drop('V200001', axis=1)

    # rename columns
    cols = {col_id : fields_map[col_id]["field_desc"] for col_id in df.columns}
    df = df.rename(columns=cols)
    selected_cols = backstory_fields + [topic]

    # select only necessary columns
    df = df[selected_cols]

    # assert that ideology is the only field with value 99
    cols_99 = [col for col in df.columns if (df[col] == 99).any()]
    assert len(cols_99) == 0 or (len(cols_99) == 1 and cols_99[0] == 'ideology')

    # remove missing data
    #df = df[((df >= 0) & (df != 99)).all(axis=1)]
    df = df[((df >= 0)).all(axis=1)]

    personas = []

    for index, row in df.iterrows():
        features_map = row.to_dict()
        persona = {}
        answers_cod = {}
        
        for topic, ans_code in features_map.items():
            ans_code = int(ans_code)
            topic_answer = None
            template = field_desc_map[topic]['template']
            
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

            answer_map = field_desc_map[topic]['valmap']
            answer_text = answer_map[topic_answer] if topic != 'age' else str(topic_answer)
            template = template.replace('XXX', answer_text)
            persona[topic] = template
            answers_cod[topic] = topic_answer

        personas.append((persona, answers_cod))

    return df, personas


def split_data(df, personas, train_ratio=0.8, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    assert len(df) == len(personas)
    n_samples = len(df)
    n_train = int(n_samples * train_ratio)
    all_indices = list(range(n_samples))
    random.shuffle(all_indices)

    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]
    df_train = df.iloc[train_indices].reset_index(drop=True)
    df_val = df.iloc[val_indices].reset_index(drop=True)
    personas_train = [personas[i] for i in train_indices]
    personas_val = [personas[i] for i in val_indices]
    return df_train, df_val, personas_train, personas_val


def k_fold_split_data(df, personas, n_folds=3, random_state=None, shuffle=True):
    assert len(df) == len(personas), 'Length of dataframe and personas must match'
    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    results = []
    
    for train_indices, val_indices in kf.split(df):
        df_train = df.iloc[train_indices].reset_index(drop=True)
        df_val = df.iloc[val_indices].reset_index(drop=True)

        personas_train = [personas[idx] for idx in train_indices]
        personas_val = [personas[idx] for idx in val_indices]
        
        results.append((df_train, df_val, personas_train, personas_val))
    return results


if __name__ == '__main__':
    df, personas = get_data()
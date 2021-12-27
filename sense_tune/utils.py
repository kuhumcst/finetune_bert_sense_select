import pandas as pd
import re

def process(s):
    if type(s) != str:
        print(s)
    if '||' in s:
        return s.split('||')
    else:
        return [s]


def clean_ontology(ontology: str) -> set:
    if type(ontology) != str:
        return set()
    return set(ontology.lower().strip('()').split('+'))


def clean_frame(frame: str) -> set:
    if type(frame) != str:
        return set()
    return set(frame.lower().split(';'))


def clean_ddo_bet(ddo_bet: str) -> int:
    if type(ddo_bet) is float:
        return 0
    if any(char.isdigit() for char in ddo_bet):
        sense = re.sub('[^0-9]', '', ddo_bet)
        if sense == '':
            return 0
        return int(sense)
    else:
        return 0

def get_fig_value(fig1, fig2):
    if fig1 == 1:
        #if fig2 == 1:
        #    return 2
        #else:
        #   return 1
        return 1
    elif fig2 == 1:
        return 1
    else:
        return 0


def get_main_sense(sense):
    if any(char.isdigit() for char in sense) and 'word2vec' not in sense:
        sense = re.sub('[^0-9]', '', sense)
        if sense == '':
            return 0
        return int(sense) * 2
    else:
        return 0

def load_datapoints_from_path(path, dataset):
    # loads and precrocess the data
    if not path:
        return None

    if dataset == 'sense_select':

        datapoints = pd.read_csv(path, sep='\t', index_col=0)
        datapoints['examples'] = datapoints['examples'].apply(process)
        datapoints['senses'] = datapoints['senses'].apply(process)
        # datapoints['target'] = datapoints['target'].apply(process)
        return datapoints

    elif dataset == 'wic':
        datapoints = pd.read_csv(path, sep='\t')
        return datapoints

    else:
        return None

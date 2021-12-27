import sense_tune.utils as utils
from collections import namedtuple
from scipy.spatial.distance import cosine


bert_data = [['lemma', 'ordklasse', 'homnr', 'bet_1', 'bet_2', 'label', 'score'],
             ['A', 'sb.', '1.', 'definition1', 'definition2', 'label', 'score']]

def predict_reduction_bert(row):
    output_data = []




"""
def predict_reduction_all(data, infotypes):
    output_data = []

    DataInstance = namedtuple('DataInstance',
                              ['lemma', 'ordklasse', 'homnr', 'bet_1', 'bet_2', 'label', 'score'])

    sense = namedtuple('sense', ['lemma', 'ordklasse', 'cor', 'ddo_bet', 'vector',
                                 'onto', 'frame', 'score', 'figurative']
                       )

    data = utils.load_datapoints_from_path(data)

    #todo: check ['lemma', 'ordklasse', 'homnr'] is in data.coloums

    for name, group in data.groupby(['lemma', 'ordklasse', 'homnr']):
        groupset = []

        for row in group.itertuples():
            figurative = row.bemaerk if type(row.bemaerk) != float else ''

            groupset.append(sense(lemma=row.lemma,
                                  ordklasse=row.ordklasse,
                                  cor=row.cor,
                                  ddo_bet=row.ddo_nr,
                                  #vector=bert,
                                  onto=utils.clean_ontology(row.onto1) \
                                  .union(utils.clean_ontology(row.onto2)),
                                  frame=utils.clean_frame(row.frame),
                                  score=int(row.t_score),
                                  figurative=1 if 'ofø' in figurative else 0))


        # pair senses and their information:
        for indx, sam1 in enumerate(groupset):
            for sam2 in groupset[indx + 1:]:
                onto_len = len(sam1.onto.intersection(sam2.onto))
                frame_len = len(sam1.frame.intersection(sam2.frame))

                point = [sam1.lemma, sam1.wcl, name[3], sam1.ddo_bet, sam2.ddo_bet]

                if 'cosine' in infotypes:
                    point.append(cosine(sam1.vector, sam2.vector))
                if 'onto' in infotypes:
                    point.append(2 if onto_len == len(sam1.onto) else 1 if onto_len >= 1 else 0)
                if 'frame' in infotypes:
                    point.append(2 if frame_len == len(sam1.frame) else 1 if frame_len >= 1 else 0)
                if 'main_sense' in infotypes:
                    point.append(1 if sam1.main_sense == sam2.main_sense else 0)
                if 'figurative' in infotypes:
                    point.append(utils.get_fig_value(sam1.figurative, sam2.figurative))

                point.append(1 if sam1.cor == sam2.cor else 0)
                output_data.append(DataInstance(point))

    return output_data
"""
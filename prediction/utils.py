import ast
import numpy as np

from data.utils import getLabelFromNumber, getApproxScoreFromLabelNumber

def to_tuple(t):
    return ast.literal_eval(t)

def to_float(a):
    return np.array(a[1:-1].split(',')).astype(float)

def generateScoresFromPrediction(predictionTotalScores):
    scores = []
    for i in predictionTotalScores:
        scores.append({
            'labelNumber': int(i['label']),
            'label': getLabelFromNumber(i['label']),
            'score': int(getApproxScoreFromLabelNumber(i['label'])),
            'roi': i['roi']
        })
    return scores
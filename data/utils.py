import json, bson
from os import name
from datetime import datetime
import numpy as np

def fixJSON(x): 
  return json.loads(json.dumps(x, cls=DateTimeEncoder))

def getLabelFromNumber(num):
  result = ''
  if num == 0: 
    result = 'omitted'
  elif num == 1:
    result = 'distorted'
  elif num == 2: 
    result = 'misplaced'
  elif num == 3:
    result = 'correct'
  else:
    result = 'unknown'
  return result

def getApproxScoreFromLabelNumber(num):
  result = 0
  if num == 0 or num == 1:
    result = num
  elif num == 2:
    result = 1
  elif num == 3:
    result = 2
  else:
    result = 0
  return result

class DateTimeEncoder(json.JSONEncoder):
  def default(self, o):
    if isinstance(o, datetime):
      return o.isoformat()
    if isinstance(o, bson.ObjectId):
      return str(o)
    if isinstance(o, np.uint8) or isinstance(o, np.int64):
      return int(o)
    if isinstance(o, np.ndarray):
      return o.tolist()
    return json.JSONEncoder.default(self, o)

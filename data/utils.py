import json, bson
from datetime import datetime

def fixJSON(x): 
  return json.loads(json.dumps(x, cls=DateTimeEncoder))

class DateTimeEncoder(json.JSONEncoder):
  def default(self, o):
    if isinstance(o, datetime):
      return o.isoformat()
    if isinstance(o, bson.ObjectId):
      return str(o)
    return json.JSONEncoder.default(self, o)

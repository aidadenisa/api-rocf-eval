

from flask_pymongo.wrappers import MongoClient
from flask_pymongo import PyMongo
from pymongo import mongo_client
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.environ.get('MONGO_URI')

def initDB(app): 
  mongoClient = PyMongo(app, uri=MONGO_URI)
  return mongoClient
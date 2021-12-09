import os
import base64
import cv2
import json
from bson import ObjectId

import redis
from rq import Queue
import time

#TODO: in production we need a pass
r = redis.Redis()
q = Queue(connection=r)

from flask import Flask, request
from flask_restful import Api, Resource, reqparse, abort, fields, marshal_with, inputs
from flask_cors import CORS, cross_origin

from preprocessing import homography
from prediction import predict_complex_scores, predict_simple_scores, model_storage, utils
from data import database
from data.utils import fixJSON

app = Flask(__name__)
api = Api(app)
CORS(app)

#DB
mongoClient = database.initDB(app)
db = mongoClient.db

app.config['CORS_HEADERS'] = 'Content-Type'

preprocessing_post_args = reqparse.RequestParser()
preprocessing_post_args.add_argument("imageb64", type=str, help="Image is missing", required=True)
preprocessing_post_args.add_argument("points", type=list, location="json", help="Image is missing", required=False)
preprocessing_post_args.add_argument("increaseBrightness", type=bool, help="Brightness is missing", required=False)
preprocessing_post_args.add_argument("gamma", type=float, help="Gamma is missing", required=False)
preprocessing_post_args.add_argument("threshold", type=int, help="Threshold is missing", required=False)

prediction_post_args = reqparse.RequestParser()
prediction_post_args.add_argument("patientCode", type=str, help="Patient code is missing", required=True)
prediction_post_args.add_argument("points", type=list, location="json", help="Image is missing", required=True)
prediction_post_args.add_argument("date", type=inputs.datetime_from_iso8601, help="Datetime is missing", required=True)
prediction_post_args.add_argument("imageb64", type=str, help="Image is missing", required=True)

revision_post_args = reqparse.RequestParser()
revision_post_args.add_argument("_rocfEvaluationId", type=str, help="Evaluation code is missing", required=True)
revision_post_args.add_argument("revisedScores", type=list, location="json", help="Revised scores are missing", required=True)
revision_post_args.add_argument("scores", type=list, location="json", help="Updated scores are missing", required=True)

revision_response = {
    "_id": fields.Raw(),
    "_rocfEvaluationId": fields.Raw(),
    "scores": fields.List(fields.Raw()),
    "revisedScores": fields.List(fields.Raw()),
}


# TODO: to be deleted

# def abort_id_not_found(name): 
#     if name not in videos:
#         abort(404, message="Video ID is not valid")

# resource fields for serialization: what gets returned from a video model
# if there are no values for this, the marshal will set them to 'None'
# if args['name']: will be false if args['name'] = 'None'
resource_fields = {
    'id': fields.Integer,
    'name': fields.String,
    'likes': fields.Integer,
    'views': fields.Integer,
}

homography_fields = {
    # 'points': fields.List, 
    'image': fields.String,
}

prediction_response = {
    "_id": fields.Raw(),
    "patientCode": fields.String(),
    "date": fields.DateTime(),
    "points": fields.Raw(),
    "scores": fields.List(fields.Raw()),
}

revision_response = {
    "_id": fields.Raw(),
    "_rocfEvaluationId": fields.Raw(),
    "revisedScores": fields.List(fields.Raw()),
}

'''
# TODO: to be deleted
# these type of classes are resources that are used for 
class HelloWorld(Resource):

    # define the serialization of the resource
    @marshal_with(resource_fields)
    # when a get request comes
    def get(self, name):
        # abort_id_not_found(name)
        # return videos[name]

        # get videos from DB, this needs to be serialized
        result = VideoModel.query.filter_by(name = name).first()
        if not result:
            abort(404, message="Could not find video with this name")
        return result

    @marshal_with(resource_fields)
    def post(self, name):
        # missing: abort if video exists
        args = video_put_args.parse_args()
        result = VideoModel.query.filter_by(name=args['name']).first()
        if result:
            abort(409, message="Video already exists")
        video = VideoModel(name=args['name'], likes=args['likes'], views=args['views'])
        db.session.add(video) # this is needed just on create, not on update
        db.session.commit()
        return video, 201
    def put(self, name):
        # missing: abort if video doesn't exist

        # check the body of the request to have all the args 
        # if missing, an error message is sent 
        args = video_put_args.parse_args()

        videos[name] = args
        
        # send back result and status code to client
        return  videos[name], 201
'''

class Preprocessing(Resource):
    @cross_origin()
    @marshal_with(homography_fields)
    def post(self):
        args = preprocessing_post_args.parse_args()
        img = homography.convertImageB64ToMatrix(args['imageb64'])

        # img = homography.emphasiseColor(img, 1.1, -50)
        imgb64 = homography.convertImageFromMatrixToB64(img)
        result = {}
        result["image"] = imgb64
        return result
class Prediction(Resource):
    @cross_origin()
    # @marshal_with(prediction_response)
    def post(self):
        # Predicts on an already binarized image
        args = prediction_post_args.parse_args()
        img = homography.convertImageB64ToMatrix(args['imageb64'])

        
        #For testing purposes, you can use this: 
        # predictionComplexScores = {
        #     "names": "Immagini-01",
        #     "scores": [305.1994, 19.02158, 69.96901, 144.1374, 40.056805, 11.410182, 24.091259],
        #     "distances": [41.23105625617661, 56.568542494923804, 10.0, 91.92388155425118, 53.85164807134504, 31.622776601683793, 36.05551275463989],
        #     "rect": [(334, 159, 54, 254), (782, 327, 87, 86), (607, 383, 230, 151), (792, 169, 268, 312), (399, 250, 123, 156), (350, 555, 150, 155), (482, 510, 308, 121)]
        # }
        
        predictionComplexScores = predict_complex_scores.predictComplexScores(img, args['points'])
        predictionTotalScores = predict_simple_scores.predictScores(img, args['points'], predictionComplexScores)
        scores = utils.generateScoresFromPrediction(predictionTotalScores)

        # test = [2, 1, 1, 0, 0, 3, 3, 1, 2, 1, 3, 1, 1, 3, 2, 3, 1, 1]

        result = {}
        result["scores"] = scores
        result["patientCode"] = args["patientCode"]
        result["date"] = args["date"]
        result["points"] = args["points"]

        insertResult = db.rocf.insert_one(result)
        # print(insertResult)
        result["_id"] = str(insertResult.inserted_id)
        return result

    @cross_origin()
    @marshal_with(prediction_response)
    def put(self):
        # Predicts on an already binarized image
        args = prediction_post_args.parse_args()

        result = {}
        result["patientCode"] = args["patientCode"]
        result["date"] = args["date"]
        result["points"] = args["points"]

        insertResult = db.rocf.insert_one(result)
        result["_id"] = str(insertResult.inserted_id)

        job = q.enqueue(ROCFevaluate, args, insertResult)

        return result
class ROCFEvaluation(Resource): 
    def get(self, id):
        #use 1 for accending, -1 for decending
        result = db.rocf.find_one({'_id': ObjectId(id)})
        return fixJSON(result), 200

class ROCFEvaluationsList(Resource): 
    def get(self):
        #use 1 for accending, -1 for decending
        result = list(db.rocf.find().sort('date', -1).limit(20))
        return fixJSON(result), 200


class ROCFRevisions(Resource): 
    # @marshal_with(revision_response)
    def post(self):
        args = revision_post_args.parse_args()
        evaluationId = args['_rocfEvaluationId']
        queryInitialEvaluation = db.rocf.find_one({'_id': ObjectId(evaluationId)})
        result = {}
        if queryInitialEvaluation: 
            result["_rocfEvaluationId"] = ObjectId(evaluationId)
            result["revisedScores"] = args["revisedScores"]

            insertResult = db.rocfRevisions.insert_one(result)
            result["_id"] = str(insertResult.inserted_id)

            updateROCF = db.rocf.update_one(
                filter = { '_id': ObjectId(evaluationId)},
                update = { '$set': {
                    'scores': args["scores"]
                    }
                }
            )
        
        return fixJSON(result), 200


# TODO: to be deleted
# api.add_resource(HelloWorld, "/helloworld/<string:name>")
# api.add_resource(HelloWorld,'/api/hello/world',endpoint='world',methods=['GET'])


def ROCFevaluate(args, DBobject):
    img = homography.convertImageB64ToMatrix(args['imageb64'])
    # For testing purposes, you can use this: 
    # predictionComplexScores = {
    #     "names": "Immagini-01",
    #     "scores": [305.1994, 19.02158, 69.96901, 144.1374, 40.056805, 11.410182, 24.091259],
    #     "distances": [41.23105625617661, 56.568542494923804, 10.0, 91.92388155425118, 53.85164807134504, 31.622776601683793, 36.05551275463989],
    #     "rect": [(334, 159, 54, 254), (782, 327, 87, 86), (607, 383, 230, 151), (792, 169, 268, 312), (399, 250, 123, 156), (350, 555, 150, 155), (482, 510, 308, 121)]
    # }
    
    predictionComplexScores = predict_complex_scores.predictComplexScores(img, args['points'])
    predictionTotalScores = predict_simple_scores.predictScores(img, args['points'], predictionComplexScores)

    #TESTT
    # predictionTotalScores = [2, 1, 1, 0, 0, 3, 3, 1, 2, 1, 3, 1, 1, 3, 2, 3, 1, 1]

    scores = utils.generateScoresFromPrediction(predictionTotalScores)

    insertResult = db.rocf.update_one(
        filter = { '_id': DBobject.inserted_id },
        update = { '$set': {
            'scores': scores
            }
        }
    )


# good
api.add_resource(Preprocessing, "/preprocessing")
api.add_resource(Prediction, "/prediction")
api.add_resource(ROCFEvaluationsList, "/rocf")
api.add_resource(ROCFEvaluation, "/rocf/<string:id>")
api.add_resource(ROCFRevisions, "/revision")


env = os.environ.get('FLASK_ENV')

if __name__ == '__main__':
    if env == 'development':
        # for production debug=False
        app.run(debug=True)
    elif env == 'production':
        app.run()

# print ("DOWNLOADING FILES")
# model_storage.downloadModels()
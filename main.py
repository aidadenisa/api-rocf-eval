import os
import base64
import cv2
import json
from bson import ObjectId
import bcrypt
import jwt
from functools import wraps

import redis
from rq import Queue
from datetime import datetime, timedelta

#TODO: in production we need a pass

r = redis.Redis(host=os.environ.get('REDIS_HOST'), port=os.environ.get('REDIS_PORT'))
q = Queue(connection=r, default_timeout=400)

from flask import Flask, request, send_from_directory, jsonify
from flask_restful import Api, Resource, reqparse, abort, fields, marshal_with, inputs
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename


from preprocessing import homography, thresholding
from prediction import predict_complex_scores, predict_simple_scores, model_storage, utils, predict_classification
from data import database, files
from data.utils import fixJSON

app = Flask(__name__)
api = Api(app)
CORS(app)

#DB
db = database.initDB(app)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_EXTENSIONS'] = ['.jpg']
app.config['UPLOAD_PATH'] = './uploads/rocfs'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')

# TESTT stuff - For testing purposes, you can use this: 
# predictionComplexScores = {
#   'names':'newImage',
#   'scores':[9.991172, 38.42955, 20.43275, 21.475847, 33.512924, 21.87283, 19.576313],
#   'distances':[55.0, 20.0, 10.0, 50.0, 14.142135623730951, 36.05551275463989, 64.03124237432849],
#   'rect':[(284, 129, 84, 274), (762, 287, 87, 86), (617, 373, 230, 151), (892, 199, 258, 302), (369, 290, 163, 196), (330, 545, 150, 155), (452, 500, 308, 121)]
# }
# predictionTotalScores = [{'label': 3, 'roi': [[[817, 585], [841, 604], [878, 556], [385, 173], [348, 221], [817, 585]], [[372, 498], [346, 514], [378, 565], [906, 235], [874, 184], [372, 498]]]}, {'label': 1, 'roi': [[[353, 188], [896, 187], [882, 560], [353, 565]], [[413, 231], [836, 228], [822, 517], [413, 524]]]}, {'label': 2, 'roi': [[[399, 250], [522, 250], [522, 406], [399, 406]]]}, {'label': 0, 'roi': []}, {'label': 1, 'roi': [[[334, 159], [388, 159], [388, 413], [334, 413]]]}, {'label': 3, 'roi': [[[388, 357], [358, 359], [363, 419], [873, 377], [903, 375], [898, 315], [388, 357]]]}, {'label': 2, 'roi': [[[718, 209], [723, 179], [664, 169], [595, 542], [590, 572], [649, 582], [718, 209]]]}, {'label': 1, 'roi': [[[657, 221], [652, 210], [647, 203], [639, 199], [628, 197], [424, 213], [413, 216], [404, 223], [398, 233], [397, 245], [400, 256], [405, 263], [472, 332], [482, 339], [490, 341], [634, 355], [648, 350], [658, 340], [661, 331], [669, 291], [669, 280], [657, 221]]]}, {'label': 0, 'roi': []}, {'label': 2, 'roi': [[[782, 327], [869, 327], [869, 413], [782, 413]]]}, {'label': 0, 'roi': []}, {'label': 1, 'roi': [[[607, 383], [837, 383], [837, 534], [607, 534]]]}, {'label': 1, 'roi': [[[916, 486], [917, 516], [977, 513], [962, 280], [961, 250], [901, 253], [916, 486]]]}, {'label': 3, 'roi': [[[1038, 319], [1032, 378], [1047, 380], [1051, 387], [1058, 393], [1065, 397], [1077, 398], [1085, 397], [1093, 392], [1114, 372], [1119, 361], [1119, 349], [1115, 338], [1107, 330], [1096, 325], [1038, 319]]]}, {'label': 2, 'roi': [[[350, 555], [500, 555], [500, 710], [350, 710]]]}, {'label': 3, 'roi': [[[830, 321], [800, 321], [801, 381], [1111, 375], [1141, 375], [1140, 315], [830, 321]]]}, {'label': 1, 'roi': [[[482, 510], [790, 510], [790, 631], [482, 631]]]}, {'label': 1, 'roi': [[[792, 169], [1060, 169], [1060, 481], [792, 481]]]}]

preprocessing_post_args = reqparse.RequestParser()
preprocessing_post_args.add_argument("imageb64", type=str, help="Image is missing", required=True)
preprocessing_post_args.add_argument("points", type=list, location="json", help="Image is missing", required=False)
preprocessing_post_args.add_argument("increaseBrightness", type=bool, help="Brightness is missing", required=False)
preprocessing_post_args.add_argument("gamma", type=float, help="Gamma is missing", required=False)
preprocessing_post_args.add_argument("threshold", type=int, help="Threshold is missing", required=False)
preprocessing_post_args.add_argument("blockSize", type=int, help="BlockSize is missing", required=False)
preprocessing_post_args.add_argument("constant", type=int, help="Constant is missing", required=False)

prediction_post_args = reqparse.RequestParser()
prediction_post_args.add_argument("patientCode", type=str, help="Patient code is missing", required=True)
prediction_post_args.add_argument("points", type=list, location="json", help="Image is missing", required=True)
prediction_post_args.add_argument("date", type=inputs.datetime_from_iso8601, help="Datetime is missing", required=False)
prediction_post_args.add_argument("imageb64", type=str, help="Image is missing", required=True)
prediction_post_args.add_argument("threshold", type=int, help="Threshold is missing", required=False)
prediction_post_args.add_argument("gamma", type=float, help="Gamma is missing", required=False)
prediction_post_args.add_argument("medium", type=str, help="Medium is missing", required=True)
prediction_post_args.add_argument("adaptiveThresholdC", type=int, help="Adaptive Threshold constant is missing", required=False)
prediction_post_args.add_argument("adaptiveThresholdBS", type=int, help="Adaptive Threshold block size is missing", required=False)

revision_post_args = reqparse.RequestParser()
revision_post_args.add_argument("_rocfEvaluationId", type=str, help="Evaluation code is missing", required=True)
revision_post_args.add_argument("revisedScores", type=list, location="json", help="Revised scores are missing", required=True)
revision_post_args.add_argument("revisedDiagnosis", type=dict, location="json", help="Revised scores are missing", required=True)
revision_post_args.add_argument("scores", type=list, location="json", help="Updated scores are missing", required=True)
revision_post_args.add_argument("diagnosis", type=dict, location="json", help="Updated scores are missing", required=True)

upload_rocf_post_args = reqparse.RequestParser()
upload_rocf_post_args.add_argument("imageb64", type=str, help="Image is missing", required=True)
upload_rocf_post_args.add_argument("patientCode", type=str, help="Patient code is missing", required=True)
upload_rocf_post_args.add_argument("doctorID", type=str, help="Doctor id is missing", required=True)
upload_rocf_post_args.add_argument("date", type=inputs.datetime_from_iso8601, help="Datetime is missing", required=True)

thresholded_homographies_post_args = reqparse.RequestParser()
thresholded_homographies_post_args.add_argument("sourceFolderURL", type=str, help="Source folder URL is missing", required=True)
thresholded_homographies_post_args.add_argument("destinationFolderURL", type=str, help="Destination folder URL is missing", required=True)
thresholded_homographies_post_args.add_argument("type", type=str, help="Type is missing", required=False)

register_post_args = reqparse.RequestParser()
register_post_args.add_argument("email", type=str, help="email is missing", required=True)
register_post_args.add_argument("name", type=str, help="name is missing", required=True)
register_post_args.add_argument("password", type=str, help="password is missing", required=True)

login_post_args = reqparse.RequestParser()
login_post_args.add_argument("email", type=str, help="email is missing", required=True)
login_post_args.add_argument("password", type=str, help="password is missing", required=True)

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

def token_required(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        token = None

        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        elif 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(' ')[1]
        
        if not token:
            return {'error':'you must be logged in'}, 401

        try:
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            # here I can query the user if I want
            current_user = db.users.find_one({'email': payload['email']})
        except:
            return {'error':'Invalid Token!'}, 401

        # return func(current_user, *args, **kwargs)
        return func(current_user, *args, **kwargs)
    
    return decorated

class Preprocessing(Resource):
    @cross_origin()
    @marshal_with(homography_fields)
    def post(self):
        # GAMMA TRANSFORM
        args = preprocessing_post_args.parse_args()
        img = homography.convertImageB64ToMatrix(args['imageb64'])

        max_color = homography.getMostFrequentColor(img)
        img = homography.removeScore(img, color=max_color)
        img = homography.adjustImage(img, gamma=args["gamma"])

        imgb64 = homography.convertImageFromMatrixToB64(img)
        result = {}
        result["image"] = imgb64
        return result
    @cross_origin()
    @marshal_with(homography_fields)
    def put(self):
        # ADAPTIVE THRESHOLDING
        args = preprocessing_post_args.parse_args()
        img = homography.convertImageB64ToMatrix(args['imageb64'])

        img = thresholding.preprocessingPhoto(img, args["points"], gamma=args["gamma"], constant= args["constant"], blockSize= args["blockSize"])

        imgb64 = homography.convertImageFromMatrixToB64(img)
        result = {}
        result["image"] = imgb64
        return result
class Prediction(Resource):
    @cross_origin()
    @token_required
    # @marshal_with(prediction_response)
    def post(current_user, self):
        # Predicts on an already binarized image
        args = prediction_post_args.parse_args()
        original = homography.convertImageB64ToMatrix(args['imageb64'])

        date = args["date"]
        if date is None:
            date = datetime.today()

        # PREPROCESSING
        threshold = 255
        if args["medium"] == "scan":
            img, threshold = thresholding.preprocessingScans(original, args["threshold"])
        elif args["medium"] == "photo":
            img = thresholding.preprocessingPhoto(original, args["points"], gamma=args["gamma"], constant= args["adaptiveThresholdC"], blockSize= args["adaptiveThresholdBS"])
        
        # HARDCODED DOCTOR ID UNTIL LOGIN
        savedFileName = files.saveROCFImages(original, img, app.config['UPLOAD_PATH'], str(current_user['_id']), args["patientCode"], date)

        # PREDICTION
        predictionComplexScores = predict_complex_scores.predictComplexScores(img, args['points'])
        predictionTotalScores = predict_simple_scores.predictScores(img, args['points'], predictionComplexScores, threshold=threshold)
        predictionDiagnosis = predict_classification.predictDiagnosis(predictionTotalScores)
        scores = utils.generateScoresFromPrediction(predictionTotalScores)

        # SAVE RESULT
        result = {}
        result["_doctorID"] = current_user['_id']
        result["scores"] = fixJSON(scores)
        result["patientCode"] = args["patientCode"]
        result["date"] = date
        result["points"] = args["points"]
        result["imageName"] = savedFileName
        result["diagnosis"] = predictionDiagnosis
        

        insertResult = db.rocf.insert_one(result)
        result["_id"] = str(insertResult.inserted_id)
        result["_doctorID"] = str(result["_doctorID"])
        return fixJSON(result)

    # THIS IS THE ENDPOINT THAT IS ACCESSED FROM THE CLIENT
    @cross_origin()
    @token_required
    @marshal_with(prediction_response)
    def put(current_user, self):
        # Predicts on an already binarized image
        args = prediction_post_args.parse_args()

        date = args["date"]
        if date is None:
            date = datetime.today()

        result = {}
        result["_doctorID"] = current_user['_id']
        result["patientCode"] = args["patientCode"]
        result["date"] = date
        result["points"] = args["points"]
        
        insertResult = db.rocf.insert_one(result)
        result["_id"] = str(insertResult.inserted_id)

        job = q.enqueue(ROCFevaluate, args, insertResult.inserted_id, current_user['_id'])

        return result
class ROCFEvaluation(Resource): 
    @token_required
    def get(current_user, self, id):
        #use 1 for accending, -1 for decending
        result = db.rocf.find_one(
            {
                '_id': ObjectId(id),
                '_doctorID': ObjectId(current_user['_id'])
            }
        )
        if result:
            return fixJSON(result), 200
        else:
            return {'error': 'ROCF not found'},404

class ROCFEvaluationsList(Resource): 
    @token_required
    def get(current_user, self):
        #use 1 for accending, -1 for decending
        result = list(db.rocf.find(
            {
                '_doctorID': ObjectId(current_user['_id']) 
            }
        ).sort('date', -1))
        return fixJSON(result), 200


class ROCFRevisions(Resource): 
    # @marshal_with(revision_response)
    @token_required
    def post(current_user, self):
        args = revision_post_args.parse_args()
        evaluationId = args['_rocfEvaluationId']
        queryInitialEvaluation = db.rocf.find_one({
            '_id': ObjectId(evaluationId),
            '_doctorID': ObjectId(current_user['_id'])
        })
        result = {}
        if queryInitialEvaluation: 
            result["_rocfEvaluationId"] = ObjectId(evaluationId)
            result["revisedScores"] = args["revisedScores"]
            result["revisedDiagnosis"] = args["revisedDiagnosis"]

            insertResult = db.rocfRevisions.insert_one(result)
            result["_id"] = str(insertResult.inserted_id)

            updateROCF = db.rocf.update_one(
                filter = { '_id': ObjectId(evaluationId)},
                update = { '$set': {
                    'scores': args['scores'],
                    'diagnosis': args['diagnosis'],
                    }
                }
            )
            return fixJSON(result), 200
        else: 
            return {'error': 'ROCF not found'}, 404

class ROCFFiles(Resource): 
    @token_required
    def get(current_user, self, docID, version, filename):
        if str(current_user['_id']) == docID:
            doctorFolderPath = os.path.join(app.config['UPLOAD_PATH'], docID, version)
            return send_from_directory(doctorFolderPath, filename)
        else:
            return {'error': 'You must be logged in to access this information'}, 401
    
    # @token_required
    # def post(self):
    #     args = upload_rocf_post_args.parse_args()
    #     files.saveROCFImage(args["imageb64"], app.config['UPLOAD_PATH'], args["doctorID"],
    #         args["patientCode"], args["date"])
    #     return 200

class ThresholdedHomographies(Resource): 
    @token_required
    def post(self):
        args = thresholded_homographies_post_args.parse_args()
        # for now, the dataset and the points can be found on the same server, in the sourcefolder
        # TODO: allow this to happen from the DB as well
        files.generateThresholdedHomographies(args["sourceFolderURL"], args["destinationFolderURL"], type=args["type"])
        return 200

class Register(Resource):
    def post(self):
        args = register_post_args.parse_args()
    
        # check if the user already exists in the DB
        existing_user = db.users.find_one({'email': args['email']})
        if existing_user is None:
            hash = bcrypt.hashpw(args['password'].encode('utf-8'), bcrypt.gensalt())
            new_user = db.users.insert_one({
                'name': args['name'],
                'email': args['email'],
                'hash': hash
            })

            # return jwt
            token = jwt.encode({
                    'email': args['email'],
                    'name': args['name'],
                    'id': str(new_user.inserted_id),
                    'expiration': str(datetime.utcnow() + timedelta(seconds=3000000))
                },
                app.config['SECRET_KEY'],
                algorithm="HS256"
            )
            return {
                'token': token,
                'email': args['email'],
                'name': args['name'],
                'id': str(new_user.inserted_id),
            }
        else:
            return {'error': 'The email address is already used'}, 400

class Login(Resource):
    def post(self):
        args = login_post_args.parse_args()

        user = db.users.find_one({'email': args['email']})
        if user is None:
            return {'error': 'The email or password is wrong'}, 401

        if bcrypt.checkpw(args['password'].encode('utf-8'), user['hash'].encode('utf-8')):
            
            # return jwt
            token = jwt.encode({
                    'email': user['email'],
                    'name': user['name'],
                    'id': str(user['_id']),
                    'expiration': str(datetime.utcnow() + timedelta(seconds=3000000))
                },
                app.config['SECRET_KEY'],
                algorithm="HS256"
            )
            return {
                'token': token,
                'email': user['email'],
                'name': user['name'],
                'id': str(user['_id']),
            }
        else:
            return {'error': 'The email or password is wrong'}, 401





def ROCFevaluate(args, rocfID, docID):
    original = homography.convertImageB64ToMatrix(args['imageb64'])

    date = args["date"]
    if date is None:
        date = datetime.today()
    
    # PREPROCESSING
    threshold = 255
    if args["medium"] == "scan":
        img, threshold = thresholding.preprocessingScans(original, args["threshold"])
    elif  args["medium"] == "photo":
        img = thresholding.preprocessingPhoto(original, args["points"], gamma=args["gamma"], constant= args["adaptiveThresholdC"], blockSize= args["adaptiveThresholdBS"])
    
    # HARDCODED DOCTOR ID UNTIL LOGIN
    savedFileName = files.saveROCFImages(original, img, app.config['UPLOAD_PATH'], str(docID), args["patientCode"], date)

    # PREDICTION
    predictionComplexScores = predict_complex_scores.predictComplexScores(img, args['points'])
    predictionTotalScores = predict_simple_scores.predictScores(img, args['points'], predictionComplexScores, threshold=threshold)
    predictionDiagnosis = predict_classification.predictDiagnosis(predictionTotalScores)
    scores = utils.generateScoresFromPrediction(predictionTotalScores)

    insertResult = db.rocf.update_one(
        filter = { '_id': rocfID },
        update = { '$set': {
            'imageName': savedFileName,
            'scores': fixJSON(scores),
            'diagnosis': predictionDiagnosis
            }
        }
    )

# good
api.add_resource(Preprocessing, "/preprocessing")
api.add_resource(Prediction, "/prediction")
api.add_resource(ROCFEvaluationsList, "/rocf")
api.add_resource(ROCFEvaluation, "/rocf/<string:id>")
api.add_resource(ROCFRevisions, "/revision")
api.add_resource(ROCFFiles, "/files", "/files/<string:docID>/<string:version>/<string:filename>")
api.add_resource(ThresholdedHomographies, "/thresholding")
api.add_resource(Register, "/register")
api.add_resource(Login, "/login")


env = os.environ.get('FLASK_ENV')

if __name__ == '__main__':
    if env == 'development':
        # for production debug=False
        # app.run(debug=False, threaded = False)
        app.run(debug=True, host='0.0.0.0') 
    elif env == 'production':
        app.run(host='0.0.0.0')

# print ("DOWNLOADING FILES")
# model_storage.downloadModels()
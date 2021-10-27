import os
import base64
import cv2

from flask import Flask, request
from flask_restful import Api, Resource, reqparse, abort, fields, marshal_with 
from flask_sqlalchemy import SQLAlchemy

from flask_cors import CORS, cross_origin

from preprocessing import homography

app = Flask(__name__)
api = Api(app)
db = SQLAlchemy(app)
CORS(app)

# TEMPORARY! REPLACE WITH THE REAL DB
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['CORS_HEADERS'] = 'Content-Type'

# TODO: to be deleted
class VideoModel(db.Model): 
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    views = db.Column(db.Integer, nullable=False)
    likes = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"Video(name={name}, views={views}, likes={likes})"

#create a database that has this model inside
db.create_all()

# TODO: to be deleted
# create a model for the objects that are sent to PUT req
video_put_args = reqparse.RequestParser()
video_put_args.add_argument("name", type=str, help="Name of the video is required", required=True)
video_put_args.add_argument("views", type=int, help="Views of the video")
video_put_args.add_argument("likes", type=int, help="Likes of the video")

homography_put_args = reqparse.RequestParser()
homography_put_args.add_argument("imageb64", type=str, help="Image is missing", required=True)
homography_put_args.add_argument("points", type=list, location="json", help="Image is missing", required=False)

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

videos = {}

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
    def delete(self, name): 
        # missing: abort if video missing

        del videos[name]
        return '', 204

class Preprocessing(Resource):
    @cross_origin()
    @marshal_with(homography_fields)
    def put(self):
        args = homography_put_args.parse_args()
        img = homography.convertImageB64ToMatrix(args['imageb64'])
        img = homography.computeHomograpy(img, args['points'])
        img = homography.removeScore(img)
        img = homography.adjustImage(img)
        # img = homography.emphasiseColor(img, 1.1, -50)
        imgb64 = homography.convertImageFromMatrixToB64(img)
        result = {}
        result["image"] = imgb64
        return result

# TODO: to be deleted
api.add_resource(HelloWorld, "/helloworld/<string:name>")

# good
api.add_resource(Preprocessing, "/preprocessing")

env = os.environ.get('FLASK_ENV')

if __name__ == '__main__':
    if env == 'development':
        # for production debug=False
        app.run(debug=True)
    elif env == 'production':
        app.run()
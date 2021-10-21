import os

from flask import Flask, request
from flask_restful import Api, Resource, reqparse, abort, fields, marshal_with 
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
api = Api(app)
db = SQLAlchemy(app)

# TEMPORARY! REPLACE WITH THE REAL DB
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'


class VideoModel(db.Model): 
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    views = db.Column(db.Integer, nullable=False)
    likes = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"Video(name={name}, views={views}, likes={likes})"

#create a database that has this model inside
db.create_all()


# create a model for the objects that are sent to PUT req
video_put_args = reqparse.RequestParser()
video_put_args.add_argument("name", type=str, help="Name of the video is required", required=True)
video_put_args.add_argument("views", type=int, help="Views of the video")
video_put_args.add_argument("likes", type=int, help="Likes of the video")

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

videos = {}

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

# resource, url
api.add_resource(HelloWorld, "/helloworld/<string:name>")

env = os.environ.get('FLASK_ENV')

if __name__ == '__main__':
    if env == 'development':
        # for production debug=False
        app.run(debug=True)
    elif env == 'production':
        app.run()
import boto3
import os
from dotenv import load_dotenv
import tempfile
import joblib
from tensorflow.keras.models import load_model

from prediction import triplet_loss

load_dotenv()

root = './' 
models_folder = root + "models/"
# models_folder='https://s3.console.aws.amazon.com/s3/object/rocf-models?region=eu-central-1&prefix='


# Creating the low level functional client
client = boto3.client(
    's3',
    aws_access_key_id = os.environ.get('S3_USER'),
    aws_secret_access_key = os.environ.get('S3_KEY'),
    region_name = 'eu-central-1'
)
resource = boto3.resource(
    's3',
    aws_access_key_id = os.environ.get('S3_USER'),
    aws_secret_access_key = os.environ.get('S3_KEY'),
    region_name = 'eu-central-1'
)
# obj = client.get_object(
#     Bucket = 'rocf-models',
#     Key = 'cross_model.joblib'
# )
# print(obj)


def retrieveModel(localPath): 
    model = None
    environment = os.environ.get('FLASK_ENV')
    # if environment == 'development':
    if environment == 'production':
        with tempfile.NamedTemporaryFile(mode='w+b') as f:
            client.download_fileobj('rocf-models', localPath, f)
            f.seek(0)
            model = joblib.load(f.name)
    elif localPath is not None: 
        model = joblib.load(models_folder + localPath)
    return model

def getKerasModel(localPath): 
    model = None
    environment = os.environ.get('FLASK_ENV')
    # if environment == 'development':
    if environment == 'production':
      # with tempfile.NamedTemporaryFile(mode='w+b') as f:
      #   client.download_fileobj('rocf-models', localPath, f, Callback=progress, Config=boto3.s3.transfer.TransferConfig(max_concurrency=50) )
      #   f.seek(0)
        model = load_model(os.path.join(models_folder, localPath), custom_objects={'batch_hard_triplet_loss': triplet_loss.batch_hard_triplet_loss, 'compute_accuracy_hard': triplet_loss.compute_accuracy})               

    elif localPath is not None: 
      model = load_model(os.path.join(models_folder, localPath), custom_objects={'batch_hard_triplet_loss': triplet_loss.batch_hard_triplet_loss, 'compute_accuracy_hard': triplet_loss.compute_accuracy})               

    return model

def progress(progress):
  print(progress)

def downloadModels(): 
  list=client.list_objects(Bucket='rocf-models')['Contents']
  for key in list:
    print("downloading model: " + key['Key'])
    client.download_file('rocf-models', key['Key'], './models/' + key['Key'])
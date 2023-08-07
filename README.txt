# ROCF Eval API

Install dependencies by running

```
pip install -r requirements.txt
```

Run the Flask REST API server by running

```
FLASK_APP=main.py FLASK_ENV=development flask run
```

In another terminal (in the same location), run a RQ worker by typing
```
rq worker
```


A few mentions about this API: 
- You should read documentation about Flask and Flask Rest Api to understand how the API works
- Everything starts from main.py; there are all the resources defined, the authorization and authentication checks, and the parsers for the call arguments. In order to make a successful call, look at the URI to access that resource (e.g. /login), the type of HTTP call (e.g. GET), and the parsers and see what arguments you need to add in the call body (e.g. login_post_args). In the body, you provide the arguments in a JSON format (e.g. { "email": "test@test.com", "password": "123456"}). You can make these requests using a HTTP client such as Postman or curl.
- Most of the resources (evaluation, preprocessing etc.) can be accessed only if you are logged in, so you first have to register a new user by making a call to /register, and then you login by making a call to /login. You will receive a JWT token as response, that you have to copy and add it in the header of the HTTP calls. The header key-value pair is 'x-access-token' : 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJlbWFpb.......' or 'Authorization' : 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJlbWFpb.......'. This gets evaluated by the function token_required and if the login is successful, the calls can be executed. 
- If you make a GET call on Prediction resource, with the appropriate arguments, you can execute prediction in batch over a set of thresholded homographies. Useful to predict on a dataset. The results get saved in total_scores.csv. 

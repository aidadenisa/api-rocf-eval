# This configuration file is used to create the Docker image of the application.

# start by pulling the python image
FROM python:3.7-slim-buster

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 gcc g++ -y

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
# RUN apk add --no-cache --update \
#     python3 python3-dev gcc \
#     gfortran musl-dev g++
RUN pip3 install -r requirements.txt

# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
# ENTRYPOINT [ "flask", "run" ]

EXPOSE 5000

CMD ["flask", "run", "-h", "0.0.0.0" ]

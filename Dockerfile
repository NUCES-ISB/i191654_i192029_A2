# init a base image (Alpine is small Linux distro)
FROM python:3.10
# update pip to minimize dependency errors 
RUN pip install --upgrade pip
# define the present working directory
WORKDIR /model
# copy the contents into the working dir
COPY . .
# run pip to install the dependencies of the flask app
RUN pip install -r requirements.txt
#train the model
RUN python stock.py
# define the command to start the container
CMD ["python","infer.py"]
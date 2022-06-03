FROM python:3.8.10
RUN mkdir /app
COPY . /app
WORKDIR /app
RUN apt update
RUN apt install libgl1 -y
RUN pip install -r requirements.txt
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
CMD python app.py
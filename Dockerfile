FROM jupyter/scipy-notebook


RUN mkdir /home/my-model/
ENV MODEL_DIR= /home/my-model/
ENV MODEL_FILE=svc_model.model

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY train.py ./train.py
#COPY api.py ./api.py
COPY app.py ./app.py

#USER 1001
EXPOSE 8080

RUN python3 train.py
CMD ["python3", "app.py", "8080"]

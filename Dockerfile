FROM python:3.8

RUN apt-get update && apt-get upgrade -y && apt-get clean && apt-get install libglu1-mesa-dev -y

RUN mkdir /root/app
COPY app /root/app
WORKDIR /root/app

EXPOSE 8501

RUN pip3 install "poetry==1.1.14"
RUN poetry install --no-dev --no-root
ENTRYPOINT [ "poetry", "run", "streamlit", "run" ]
CMD [ "app.py" ]



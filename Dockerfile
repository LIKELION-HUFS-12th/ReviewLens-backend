FROM python:3.10.12

ENV PYTHONUNBUFFERED 1

RUN apt-get -y update
RUN apt-get -y install vim

RUN mkdir /app
ADD . /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY project/local_settings.py .

CMD ["gunicorn", "project.wsgi:application", "--bind", "0.0.0.0:8000", "--workers=3", "--timeout=120"]

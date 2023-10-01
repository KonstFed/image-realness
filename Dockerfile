FROM python:3.10

COPY . /app
WORKDIR /app

RUN pip3 install -r freeze_requirements.txt

EXPOSE 5000
CMD ["python3", "app.py"]
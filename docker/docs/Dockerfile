FROM python:3.7-slim-buster

# Install make for docs and libsndfile for librosa
RUN apt-get update && \
    apt-get install --reinstall make -y && \
    apt-get install wget libsndfile1 -y

# Install documentation packages
COPY . /opt/audiolib
RUN pip install -U pip && \
    pip install -r /opt/audiolib/requirements.txt --no-deps --ignore-installed

WORKDIR /opt/audiolib/docs
RUN make html
WORKDIR /opt/audiolib/docs/build/html
ENTRYPOINT ["python", "-u", "-m", "http.server", "8080"]
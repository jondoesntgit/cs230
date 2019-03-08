FROM continuumio/miniconda3

COPY requirements.txt /requirements.txt
COPY .env /.env
COPY Makefile /Makefile

RUN conda create -n env python=3.6 &&  echo "source activate env" > ~/.bashrc
CMD set PYTHONIOENCODING=utf-8

RUN echo "deb http://http.us.debian.org/debian sid main non-free contrib" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y ffmpeg build-essential libsndfile-dev

RUN pip install -r requirements.txt

RUN make vggish_params
COPY src /src

CMD cd /src/utils
ENTRYPOINT cd /src/utils && python db2vggish.py

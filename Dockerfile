FROM continuumio/miniconda3
RUN conda create -n env python=3.6
RUN echo "source activate env" > ~/.bashrc
CMD set PYTHONIOENCODING=utf-8

RUN echo "deb http://http.us.debian.org/debian sid main non-free contrib" >> /etc/apt/sources.list
RUN apt-get update

ADD requirements.txt /requirements.txt
ADD .env /.env

RUN apt-get install -y libsndfile-dev
RUN pip install -r requirements.txt

RUN apt-get install -y build-essential
ADD Makefile /Makefile
RUN make vggish_params

ADD src /src

RUN pip install requests
RUN apt-get install -y ffmpeg

CMD cd /src/utils/ && python db2vggish.py

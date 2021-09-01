FROM yankovg/python3.8.2-ubuntu18.04

RUN apt-get update
RUN apt install git -y

RUN git clone https://github.com/Wangsherpa/mushroom_identification_app.git

RUN python3.8 -m pip install --upgrade pip
RUN pip install streamlit==0.85.0
RUN pip install numpy==1.20.3
RUN pip install matplotlib==3.4.2
RUN pip install seaborn==0.11.1
RUN pip --default-timeout=10000 install tensorflow==2.4.0c0

RUN apt-get install libbz2-dev -y

WORKDIR mushroom_identification_app/

CMD streamlit run app.py
CMD sleep 1000

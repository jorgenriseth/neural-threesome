# syntax=docker/dockerfile:1
FROM ceciledc/fenics_mixed_dimensional:latest
RUN apt-get update
RUN apt-get install libglu1-mesa libxcursor1 libxft2 libxinerama1 -y

WORKDIR /home/fenics
RUN wget https://gmsh.info/bin/Linux/gmsh-4.10.5-Linux64.tgz --no-check-certificate
RUN tar xvf gmsh-4.10.5-Linux64.tgz && rm gmsh-4.10.5-Linux64.tgz
ENV PATH="${PATH}:/home/fenics/gmsh-4.10.5-Linux64/bin"

RUN mkdir /home/fenics/neural-threesome
WORKDIR /home/fenics/neural-threesome
COPY ./requirements.txt requirements.txt
COPY ./setup.py setup.py 
COPY ./neuralthreesome/__init__.py neuralthreesome/__init__.py

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install -e . 

EXPOSE 8080

FROM jupyter/base-notebook
FROM python:3.8.16
USER root
RUN apt-get update -y
RUN apt install libgl1 libgl1-mesa-glx libglib2.0-0 -y

# Install Python 3.8 using conda
#RUN conda install --quiet --yes python=3.8

# Create a new Conda environment and activate it
#RUN conda create --quiet --yes --name myenv python=3.8
#RUN echo "conda activate myenv" >> ~/.bashrc

# Set the default working directory
WORKDIR /home/jovyan

# Install additional packages ( from requirements.txt file)
#COPY requirements.txt /home/jovyan/
COPY prediction_file.ipynb /home/jovyan/
COPY Image /home/jovyan/
COPY Results /home/jovyan/
COPY model_architecture.h5 /home/jovyan/
COPY spore_test_dec5.hdf5 /home/jovyan/
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install matplotlib>=3.2.2  
RUN pip install pillow>=7.1.2  
RUN pip install pandas>=1.1.4 
RUN pip install pyyaml>=5.3.1  
RUN pip install seaborn>=0.11.0 
RUN pip install requests>=2.23.0 
RUN pip install scipy==1.10.1
RUN pip install tensorflow==2.7.0
RUN pip install keras==2.7.0
RUN pip install h5py==3.10.0
RUN pip install opencv-python==4.9.0.80 
RUN pip install tqdm==4.66.1 
RUN pip install scikit-image==0.21.0 
RUN pip install scikit-learn==1.3.2




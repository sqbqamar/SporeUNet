FROM jupyter/base-notebook
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
#COPY Results /home/jovyan/
COPY model_architecture.h5 /home/jovyan/
COPY spore_test_dec5.hdf5 /home/jovyan/
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install matplotlib 
RUN pip install pillow
RUN pip install pandas
RUN pip install pyyaml
RUN pip install seaborn 
RUN pip install requests 
RUN pip install scipy
RUN pip install tensorflow
RUN pip install keras
RUN pip install h5py
RUN pip install opencv-python
RUN pip install tqdm
RUN pip install scikit-image
RUN pip install scikit-learn

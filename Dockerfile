FROM jupyter/base-notebook
USER root

# Install system packages
RUN apt-get update -y
RUN apt install libgl1 libgl1-mesa-glx libglib2.0-0 -y

# Copy the environment file
COPY spore_env.yml /tmp/spore_env.yml

# Create a new Conda environment from the environment file
RUN conda env create -f /tmp/spore_env.yml

# Activate the new environment
ENV PATH /opt/conda/envs/spore/bin:$PATH

# Set the default working directory
WORKDIR /home/jovyan

# Install additional packages ( from requirements.txt file)
#COPY requirements.txt /home/jovyan/
COPY prediction_file.ipynb /home/jovyan/
COPY Image /home/jovyan/
#COPY Results /home/jovyan/
COPY model_architecture.h5 /home/jovyan/
COPY spore_test_dec5.hdf5 /home/jovyan/

# Switch back to root user
USER root

# SporeUNet
## Interactive Application using Flask API

You can deploy Flask based API in an interactive way using the following command.

`python app.py`

Open the application in any browser 0.0.0.0:5000 and upload your image. Application will return the labeled image.

## Run code with Command Prompt 
- Open `cmd_prediction.py` and set the model architecture and weight paths. then save the file. 

- Using the command line, navigate to the directory where you saved `cmd_prediction.py`. 

- Run the script by typing the following command:

`python cmd_predictor.py path/to/input/image.jpg path/to/output/directory`

Replace `path/to/input/image.jpg` with the path to the image you want to process and `path/to/output/directory` with the path to the directory where you want to save the output. 


## Run code with Binder
After clicking on launch button, Binder environment will automatically configure required libraries and packages. This may take a few moments to complete   

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sqbqamar/SporeUNet/master?labpath=prediction_file.ipynb)

- Binder will open the `prediction_file.ipynb` notebook
- Check model architecture , weight, and image paths. 
- Notebook contains cell-by-cell code
- Examine each cell's code and comments
- Understand how model is run for predictions

## Train SporeUNet with Custom Dataset
Just open the `Training_model.ipynb` file on Google Colab or the local system and follow the instructions as written in the file.  
[Open Training file in Google Colab] (https://colab.research.google.com/github/sqbqamar/SporeUNet/blob/main/Training_model.ipynb)

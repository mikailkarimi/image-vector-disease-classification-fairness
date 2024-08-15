# Image and Vector Disease Classification Fairness

EECS 4080 SU 2024 Project with supervisor Dr. Laleh Seyyed-Kalantari.

All configuration files are located in the ./configs directory. Before running the training or testing commands, adjust the parameters in the .yaml files according to your requirements.

## Training
To train your model you can use the following command. Replace CONFIG_FILE with the name of your specific configuration file.

`python main.py fit -c ./configs/CONFIG_FILE`

## Testing
After training, depending on your use case, use the non-graphing notebooks found in the notes directory to obtain data.

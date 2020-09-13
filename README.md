# Number_Recognition_Neural_Network

## What is this?

This is my implementation of the back-propogation algorithm to classify handwritten digits from 0-9, without an external machine learning library. It utillises both a command line and gui interface.

## How to use this?

#### Training
1. In display.py ensure that the following code is commented out:
```
while True:
    Initialise(True, 20, 25)
    Run()
```
2. In the number_classifier.py file ensure that ```train = True```. 
3. You can change around values such as maximum_iterations, learning_rate, lambada, and parameters. 
4. Run number_classifier.py to train

#### Testing Network
1. When the training for each class has been completed a gui will pop up. You can then draw your number to test the network and press right click when you are complete.
2. The output from the neural network will be displayed in the command line.
3. The previous steps are then repeated.

#### Adding Training Data
1. In display.py ensure that ```create_training_data = True``` and  that the following code is not commented out:
```
while True:
    Initialise(True, 20, 25)
    Run()
```
2. Run display.
3. Left click on the pixels and hold to draw your number.
4. Add the label by typing it into the text input box (hint: you can go into ui.py and change the default text value for the input box to avoid this step). 
4. Right click to save the data.


#### Editing Network Shape
The parameters are stored as lists of numpy arrays, where each array is the weights associated with each layer. You can edit the number of layers by adding or deleting an array. The number of neurons in each layer are the first numpy.array.shape value and the second value represent the number of inputs that that layer will have (i.e how many neurons / input values there were in the previous layer).The below example cuts the number of neurons in the first layer by half.
```
parameters = [np.random.normal(0, 1, size= (200, 625))...
```
to
```
parameters = [np.random.normal(0, 1, size= (100, 625))...
```

Do not change the amount of input data in the first layer as that is constant (625 is the number of pixels in each number) nor the number of neurons in the last layer. The below example removes a hidden layer from the network.
```
parameters = [np.random.normal(0, 1, size= (200, 625)), np.random.normal(0, 1, size= (10, 201)), np.random.normal(0, 1, size =(1, 11))]
```
```
parameters = [np.random.normal(0, 1, size= (200, 625)), np.random.normal(0, 1, size =(1, 201))]
```

When removing a layer ensure to update the number of inputs for the layer after the layer that was removed.

## Data

#### Data Description

The data associated with a single training value for a number is obtained as a 25x25 matrix and is then converted into a 1x625 vector.

#### Data Collection

With the use of the display script, data can be collected if ```create_training_data = True```. 

#### Data Storage

The digits are stored as a numpy array (in binary format) in data/x_list and the labels for the data are stored in data/y_list. There is a description of the data in the data folder which shows how much training data there is at the moment for each number. 

## File Reference

#### number_classifier.py
This file can train the neural network using the back-propogation and gradient descent algorithm or simply test the network with the already trained parameters. With the use of the number_classifier script, if ```train = True``` the network will be trained (this can take quite a while).

#### display.py
This is an inteface created using pygame which can be used to add to the training data set or test the network. Data can be collected if ```create_training_data = True``` and the following code is commented out:
```
while True:
    Initialise(True, 20, 25)
    Run()
```
When testing the network the user simply has to draw his the number and then submit it; there is no need to enter a label.

#### ui.py
This file contains my custom classes use by the display file (example: button(pixels used for drawing).

#### x_list.npy / y_list.npy
This is where the training data is stored

#### user_input.npy
This file is used to store the input data created when testing the network.

#### data_description_binary.npy / data_description_readable.txt
These files describe the data stored in the training data.

#### parameters
These are the learnt parameters for each class.

## Exterbal Modules Used
1. Numpy v1.18.5 (for mathematics)
2. Pygame v1.9.6 (for gui)
3. Pickle (for reading and writing to files)
4. Os (for reading and writing to files)
5. Sys (for geting current directory)


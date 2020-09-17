import numpy as np
import os
import sys
import pickle
import display
import time

train = False

#Input and ouput values that the neural network will train on
x_list = np.array([])
y_list = np.array([])

#weights
parameters = []

#The error calculated for a single x value
small_delta = []
#The error calculated for the entire training data
total_delta = []

#x_propogated is often denoted as the 'activation' values of each layer.
x_propogated = []

h = np.zeros((1,1))
#number of ouputs
k = 1
#number of layers
l = 0
#number of data sets
m = 7

learning_rate = 0.1
lambada = 0.0005
max_iterations = 10000 #5000


def Initialise():
    global x_propogated, m, l, k, total_delta, small_delta, y_list, x_list, parameters
    print("\nBeginning initialisation...")

    x_list = np.load(os.path.join(sys.path[0], 'data', "x_list.npy"))
    y_list = np.load(os.path.join(sys.path[0], 'data', "y_list.npy"))

    m = len(y_list)
    k = 1
    for y in y_list:
        if y > k:
            k = int(y) + 1
    print(k)

    li = np.zeros((y_list.shape[0], k))
    i = 0
    while i < len(y_list):
        li[i][y_list[i]] = 1
        i += 1
    y_list = li

    print(y_list.shape)
    #Initialises the parameters of each layer to random values.
    parameters = [np.random.normal(0, 1, size= (200, 625)), np.random.normal(0, 1, size= (100, 201)), np.random.normal(0, 1, size =(k,101))]
    l = len(parameters)

    #Creates the shape of x_propogated from the list of parameters
    for param in parameters:
        x_propogated.append(np.array(param))
    print(x_propogated[l-1].shape)
  
def Sigmoid(x, param):
    z = (np.matmul(x,  np.transpose(param)))
    return 1 / (1 + np.exp(np.negative(z)))

def Forward_Propogate(x_row, special_parameters = None):
    global x_propogated, parameters

    if not special_parameters == None:
        #Calculates the activation value of the first layer
        x_propogated[0] = Sigmoid(x_row, special_parameters[0])
        #Adds a bias unit to x_propogated (activation) so that it can be immediately used to calculate the activation of the next layer
        x_propogated[0] = np.hstack((np.ones(1), x_propogated[0]))

        i = 1
        while i < l:
            #Calculates the activation value of every subsequent layer
            x_propogated[i] = Sigmoid(x_propogated[i - 1], special_parameters[i])
            if i != l - 1:
                #If this is not the last activation value(neural network output) a bias unit is added to x_propgated
                x_propogated[i] = np.hstack((np.ones(1), x_propogated[i]))
            i += 1
        return

    #Calculates the activation value of the first layer
    x_propogated[0] = Sigmoid(x_row, parameters[0])
    #Adds a bias unit to x_propogated (activation) so that it can be immediately used to calculate the activation of the next layer
    x_propogated[0] = np.hstack((np.ones(1), x_propogated[0]))

    i = 1
    while i < l:
        #Calculates the activation value of every subsequent layer
        x_propogated[i] = Sigmoid(x_propogated[i - 1], parameters[i])
        if i != l - 1:
            #If this is not the last activation value(neural network output) a bias unit is added to x_propgated
            x_propogated[i] = np.hstack((np.ones(1), x_propogated[i]))
        i += 1

def Prevent_Error(y):
    #To prevent the divide by zero error in the cost function the vlue is changed by a little bit
    if y == 1:
        y = 0.999999
    elif y == 0:
        y = 0.000001
    return y

def Regularise_Parameters(parameters, sum_ = True):
    global lambada

    if not sum_:
        list_ = []
        for parameter in parameters:
            list_.append(parameter * lambada)

        return list_

    r = 0
    for parameter in parameters:
        i = 0
        while i < parameter.shape[0]:
            if len(parameter.shape) > 1:
                j = 0
                while j < parameter.shape[1]:
                    r += parameter[i,j] ** 2
                    j += 1
            else:
                r += parameter[i] ** 2
            i += 1
    return r * lambada

def Loss(class_return = 0):
    global x_list, y_list, x_propogated, m, l, k, parameters, h
    #Calculates the total error of the neural network 
    '''
    i = 0
    h = []
    while i < m:
        Forward_Propogate(x_list[i])
        h.append(Prevent_Error(x_propogated[l - 1]))
        i += 1

    h = np.transpose(np.array(h)[np.newaxis])
    cost = ((np.matmul(np.negative(y_list).T, np.log(h)) - np.matmul((1 - y_list).T, np.log(1 - h))) / m)
    '''

    cost = []
    Forward_Propogate(x_list[0])
    h = x_propogated[l - 1]
    i = 1
    while i < len(x_list):
        Forward_Propogate(x_list[i])
        h = np.vstack((h, x_propogated[l-1]))
        i += 1
    #print(h.shape)
    #print(y_list.shape)
    #print(h)
    #print(y_list)
    #print(np.matmul(np.negative(y_list).T, np.log(h)))
    #print(h)
    h_ = np.transpose(np.transpose(h)[0][np.newaxis])
    y_list_ = np.transpose(np.transpose(y_list)[0][np.newaxis])
    cost = (np.matmul(np.negative(y_list_).T, np.log(h_)) - np.matmul((1 - y_list_).T, np.log(1 - h_)) + Regularise_Parameters(parameters)) / m 
    
    i = 1
    while i < k:
        h_ = np.transpose(np.transpose(h)[0][np.newaxis])
        y_list_ = np.transpose(np.transpose(y_list)[0][np.newaxis])
        cost += (np.matmul(np.negative(y_list_).T, np.log(h_)) - np.matmul((1 - y_list_).T, np.log(1 - h_)))
        i += 1
    cost /= m
    cost += Regularise_Parameters(parameters)
    return cost

def Save_Parameters(parameters):
    file = open(os.path.join(sys.path[0],"parameters"), "wb")
    pickle.dump(parameters, file)
    file.close()

def Load_Parameters():
    global parameters
    file_ = open(os.path.join(sys.path[0], "parameters"), "rb")
    parameters = pickle.load(file_)

def Back_Propogation():
    global l, x_list, y_list, small_delta, total_delta, parameters, m
    small_delta.clear()
    total_delta.clear()

    #Creates the shape of the change varables
    for param in parameters:
        small_delta.append(param)
        total_delta.append(np.zeros(param.shape))

    i = 0
    while i < m:
        #Finds x_propogated for specified x value
        Forward_Propogate(x_list[i])

        l_ = l - 1
        while l_ > -1:
            #if at last layer in neural network the error of the layer is calculated for a specific x-value
            if l_ == l - 1:
                small_delta[l_] =  x_propogated[l_] - y_list[i]
            
            #the error of each subsequent layer is calculated  for a specific x-value
            else:
                sigmoid_derivative = x_propogated[l_] * (1 - x_propogated[l_])
                sigmoid_derivative = np.delete(sigmoid_derivative, 0)
                temp = np.transpose(parameters[l_ + 1]) * small_delta[l_ + 1]
                #removing the bias unit from this array
                temp = np.delete(np.transpose(temp), 0, 1)
                if temp.shape[0] == 1:
                    temp = temp[0,...]
                if len(temp.shape) > 1:
                    if temp.shape[0] >= 1:
                        temp = np.sum(temp,0)
                small_delta[l_] = temp * sigmoid_derivative
            l_ -= 1

        l_ = l - 1
        while l_ > -1:
            #The error of the first layer is calculated using the input x values and added to the total error
            if l_ == 0:
                total_delta[l_] += np.transpose(small_delta[l_][np.newaxis]) * x_list[i]
            
            #the error of the each other layer is calculated using the propogated x values and added to the total error
            else:
                total_delta[l_] += np.transpose(small_delta[l_][np.newaxis]) * x_propogated[l_ - 1]
            l_ -= 1
            

        i += 1
    
    #At the end of back-propogation the averaged total_delta of all layers are found
    for change in total_delta:
        change /= m

def Gradient_Descent():
    global parameters, max_iterations, total_delta, h
    print("Beginning gradient descent...\n")
    
    #if the cost begins to increase the lowest possible cost and its corresponding data is recorded
    
    '''lowest_loss = 10 ** 20
    lowest_iteration = 0
    lowest_parameter = []
    for param in parameters:
        lowest_parameter.append(param)'''
    
    #Back propogates till max_iteration is reached
    i = 0
    t = time.time()
    while i < max_iterations:
        if i % 100 == 0:
            print(f"Loss: {Loss()}\tIteration: {i}")
            if not i == 0:
                print(f"Time taken: {round(time.time() - t, 2)}")
                t = time.time()
        elif i == max_iterations - 1:
            print(f"Loss: {Loss()}\tIteration: {i}")
            print(f"Time taken: {round(time.time() - t, 2)}\n")
            #print(f"Parameters: {parameters}")

        Back_Propogation()
        
        #updates parameters
        j = 0
        temp = Regularise_Parameters(parameters, False)
        while j < len(parameters):
            parameters[j] -= (learning_rate * total_delta[j]) + temp[j]
            j += 1

        i += 1

    Save_Parameters(parameters)

def TestNetwork():
    global x_propogated
    print("\nEnter anything besides a number into the command line to quit or press the x on the gui to quit.\n")
    while True:
        print("Draw number")
        display.create_training_data = False
        display.Initialise(False, 20, 25)
        display.Run()
        #exec(open(os.path.join(sys.path[0],"display.py")).read(), globals())
        x_list_input = np.load(os.path.join(sys.path[0], "data", "user_input.npy"))

        Forward_Propogate(x_list_input)
        print(f"Probablility of entered values is: {(x_propogated[l - 1])}")
        print()


if train:
    Initialise()
    Gradient_Descent()
    TestNetwork()

else:
    Initialise()
    Load_Parameters()
    TestNetwork()
'''
Initialise()
t = time.time()
print(Loss())
print(time.time() - t)
'''

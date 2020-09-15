import numpy as np
import os
import sys
import pickle
import display
import time

train = True

#Input and ouput values that the neural network will train on
x_list = np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1]])
y_list = np.array([[1], [1], [0], [0]])
new_y_list = []

#weights
parameters = []
#parameters calculated for each class
parameters_result = []

#The error calculated for a single x value
small_delta = []
#The error calculated for the entire training data
total_delta = []

#x_propogated is often denoted as the 'activation' values of each layer.
x_propogated = []
x_all_propogated = []

#number of ouputs
k = 1
#number of layers
l = 0
#number of data sets
m = 7

learning_rate = 1
lambada = 0.0005
max_iterations = 1000 #5000
minimum_loss = 0.05


def Initialise(only_param = False):
    global x_propogated, m, l, k, total_delta, small_delta, y_list, new_y_list, x_list, parameters

    #Initialises the parameters of each layer to random values.
    parameters = [np.random.normal(0, 1, size= (100, 625)), np.random.normal(0, 1, size= (10, 101)), np.random.normal(0, 1, size =(1, 11))]
    if only_param:
        return

    print("\nBeginning initialisation...")


    x_list = np.load(os.path.join(sys.path[0], 'data', "x_list.npy"))
    y_list = np.load(os.path.join(sys.path[0], 'data', "y_list.npy"))

    #Creates the shape of x_propogated from the list of parameters
    for param in parameters:
        x_propogated.append(np.array(param))
        x_all_propogated.append(np.array(param))
    
    m = len(y_list)
    l = len(parameters)

    k = 1
    for y in y_list:
        if y > k:
            k = y
        
    j = 0
    while j <= k:
        temp = []
        i = 0
        while i < m:
            temp.append(y_list[i])
            if not temp[i] == j:
                temp[i] = 0
            else:
                temp[i] = 1
            i += 1
            
        new_y_list.append(np.array(temp))
        j += 1
  
def Sigmoid(x, param):
    return 1 / (1 + np.exp(np.negative(np.matmul(x,  np.transpose(param)))))

def Forward_Propogate_All(x_row):
    global x_propogated, parameters

    #Calculates the activation value of the first layer
    x_all_propogated[0] = Sigmoid(x_row, parameters[0])
    #Adds a bias unit to x_propogated (activation) so that it can be immediately used to calculate the activation of the next layer
    x_all_propogated[0] = np.hstack((np.ones((x_all_propogated[0].shape[0], 1)), x_all_propogated[0]))

    i = 1
    while i < l:
        #Calculates the activation value of every subsequent layer
        x_all_propogated[i] = Sigmoid(x_all_propogated[i - 1], parameters[i])
        if i != l - 1:
            #If this is not the last activation value(neural network output) a bias unit is added to x_propgated
            x_all_propogated[i] = np.hstack((np.ones((x_all_propogated[i].shape[0], 1)), x_all_propogated[i]))
        i += 1

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
        r += np.sum(parameter ** 2)
    return r * lambada

def Loss(class_return = -1):
    global x_list, y_list, x_propogated, x_all_propogated, m, l, parameters
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
    class_ = 0
    while class_ <= k:
        Forward_Propogate_All(x_list)
        h = x_all_propogated[l - 1]
        cost.append((np.matmul(np.negative(new_y_list[class_]).T, np.log(h)) - np.matmul((1 - new_y_list[class_]).T, np.log(1 - h)) + Regularise_Parameters(parameters)) / m)
        class_ += 1  

    if class_return < 0:
        return cost
    else:
        return cost[class_return]

def Save_Parameters(class_, parameters):
    file = open(os.path.join(sys.path[0],"parameters", f"parameters{class_}"), "wb")
    pickle.dump(parameters, file)
    file.close()

def Load_Parameters():
    global parameters_result
    i = 0
    while i < len(os.listdir(os.path.join(sys.path[0], "parameters"))):
        file_ = open(os.path.join(sys.path[0], "parameters", f"parameters{i}"), "rb")
        parameters_result.append(pickle.load(file_))
        i += 1

def Back_Propogation(class_):
    global l, x_list, y_list, new_y_list, small_delta, total_delta, parameters, m
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
                small_delta[l_] =  x_propogated[l_] - new_y_list[class_][i]
            
            #the error of each subsequent layer is calculated  for a specific x-value
            else:
                #sigmoid deriv
                #sigmoid_derivative = x_propogated[l_] * (1 - x_propogated[l_])
                #igmoid_derivative = np.delete(sigmoid_derivative, 0)
                temp = np.transpose(parameters[l_ + 1]) * small_delta[l_ + 1]
                #removing the bias unit from this array
                temp = np.delete(np.transpose(temp), 0, 1)
                if temp.shape[0] == 1:
                    temp = temp[0,...]
                if len(temp.shape) > 1 and temp.shape[0] >= 1:
                    temp = np.sum(temp,0)
                small_delta[l_] = temp * np.delete(x_propogated[l_] * (1 - x_propogated[l_]), 0)
            l_ -= 1
        l_ = l - 1
        while l_ > -1:
            #The error of the first layer is calculated using the input x values and added to the total error
            if l_ == 0:
                total_delta[l_] += (np.transpose(small_delta[l_][np.newaxis]) * x_list[i]) / m
            
            #the error of the each other layer is calculated using the propogated x values and added to the total error
            else:
                total_delta[l_] += (np.transpose(small_delta[l_][np.newaxis]) * x_propogated[l_ - 1]) / m
            l_ -= 1

        i += 1
    
def Gradient_Descent():
    global parameters, parameters_result, max_iterations, k, total_delta
    print("Beginning gradient descent...\n")
    
    #if the cost begins to increase the lowest possible cost and its corresponding data is recorded
    
    '''lowest_loss = 10 ** 20
    lowest_iteration = 0
    lowest_parameter = []
    for param in parameters:
        lowest_parameter.append(param)'''
    
    #Back propogates till max_iteration is reached
    class_ = 0
    while class_ <= k:
        i = 0
        lowest_loss = 10000000
        Initialise(True)
        print(f"Calculating parameters for class {class_}")
        t = time.time()

        while i < max_iterations:
            cost = Loss(class_)
            if i % 20 == 0:
                print(f"Loss: {cost}\tIteration: {i}")
                if not i == 0:
                    print(f"Time taken: {round(time.time() - t, 2)}")
                    t = time.time()
                print("\n")

            elif i == max_iterations - 1:
                print(f"Loss: {cost}\tIteration: {i}")
                print(f"Time taken: {round(time.time() - t, 2)}\n")
            
            elif cost <= minimum_loss or cost > lowest_loss:
                print(f"LowestLoss: {cost}\tIteration: {i}")
                print(f"Time taken: {round(time.time() - t, 2)}\n")
                i = max_iterations

            lowest_loss = cost
            Back_Propogation(class_)
            
            #updates parameters
            j = 0
            temp = Regularise_Parameters(parameters, False)
            while j < len(parameters):
                parameters[j] -= (learning_rate * total_delta[j]) + temp[j]
                j += 1

            '''
            if Loss() < lowest_loss:
                lowest_loss = Loss()
                lowest_iteration = i
                lowest_parameter.clear()
                for param in parameters:
                    lowest_parameter.append(param)
            '''
            i += 1
        parameters_result.append(parameters)
        Save_Parameters(class_, parameters_result[class_])
        class_ += 1

    '''
    if Loss() > lowest_loss:
        print("\n\nThe final loss value obtained was greater than the lowest loss obtained.")
        print("\nLowest Loss: " + str(lowest_loss) + "\t iteration: " + str(lowest_iteration))
        print("Lowest Parameters:" + str(lowest_parameter))
        print("\n")
    '''

def TestNetwork():
    global k, l, parameters_result, x_propogated
    print("\nEnter anything besides a number into the command line to quit or press the x on the gui to quit.\n")
    while True:
        print("Draw number")
        display.create_training_data = False
        display.Initialise(False, 20, 25)
        display.Run()
        #exec(open(os.path.join(sys.path[0],"display.py")).read(), globals())
        x_list_input = np.load(os.path.join(sys.path[0], "data", "user_input.npy"))

        Forward_Propogate(x_list_input)

        probabilities = np.zeros((3,2))
        class_ = 0
        while class_ < k + 1:
            Forward_Propogate(x_list_input, parameters_result[class_])
            if x_propogated[l - 1] > probabilities[0,1]:
                probabilities[0] = np.array([class_, x_propogated[l-1]])
            elif x_propogated[l - 1] > probabilities[1,1]:
                probabilities[1] = np.array([class_, x_propogated[l-1]])
            elif x_propogated[l - 1] > probabilities[2,1]:
                probabilities[2] = np.array([class_, x_propogated[l-1]])
            class_ += 1
        print(f"Probability of input equaling {probabilities[0,0]} is {probabilities[0,1]}")
        print(f"Probability of input equaling {probabilities[1,0]} is {probabilities[1,1]}")
        print(f"Probability of input equaling {probabilities[2,0]} is {probabilities[2,1]}")
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
Loss()
print(f"lo {time.time()-t}")
t = time.time()
Back_Propogation(1)
print(f"bp {time.time()-t}")
t = time.time()
Forward_Propogate(x_list[0])
print(f"fp {time.time()-t}")
print(new_y_list[0])
'''
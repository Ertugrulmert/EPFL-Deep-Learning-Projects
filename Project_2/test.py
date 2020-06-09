# test file to use the framework of proj2

import torch
import framework as frame
import math
import matplotlib.pyplot as plt
import numpy as np


torch.set_grad_enabled(False);


####################### Helper functions for testing #########################

# Generate dataset and normalize input values
def generate_norm_data():
    samples = 1000
    torch.manual_seed(42)
    
    inputs = torch.rand(samples,2)
    distance = torch.norm((inputs - torch.Tensor([[0.5, 0.5]])), 2, 1, True)
    targets = distance.mul(math.sqrt(2*math.pi)).sub(1).sign().add(1).div(2).long()  
    
    return inputs, targets

# Convert targets to -1,1
def labels_conversion(targets, inputs):
    
    # Normalize data
    mean, std = inputs.mean(), inputs.std()
    inputs.sub_(mean).div_(std)
    
    new_targets = inputs.new_zeros((targets.size(0), targets.max() + 1)).fill_(-1)
    
    return new_targets.scatter_(1, targets.view(-1, 1), 1.0), mean, std

# Splits inputs and targets following the given repartition
def data_split(inputs, targets):
    
    #the split is inside instead of as a parameter to assure proper repartition
    train_part = 0.7
    val_part = 0.1
    test_part = 0.2

    training_size = math.floor(inputs.size()[0] * train_part)
    train_data = inputs.narrow(0, 0, training_size)
    train_targets = targets.narrow(0, 0, training_size)
    
    val_size = math.floor(inputs.size()[0] * val_part)
    validation_data = inputs.narrow(0, training_size, val_size)
    validation_targets = targets.narrow(0, training_size, val_size)
    
    test_size = math.floor(inputs.size()[0] * test_part)
    test_data = inputs.narrow(0, training_size+val_size, test_size)
    test_targets = targets.narrow(0, training_size+val_size, test_size)
    
    return train_data, train_targets, validation_data, validation_targets,\
           test_data, test_targets




############################# Testing starts #################################

# Generate dataset
inputs, targets = generate_norm_data()

# Plot datapoints and their true targets/labels
plt.figure(figsize=(6,6)) 
plt.scatter(inputs[:,0].numpy(), inputs[:,1].numpy(),  c = (np.squeeze(targets.numpy())),cmap = 'seismic',alpha=0.7)
plt.title("Ground truth Random dataset in [0,1]², ones are in blue and zeros in red")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Convert targets to labels of -1 and 1
targets, mean, std = labels_conversion(targets, inputs)



# Data split into training, validation and test 
train_data, train_targets, \
validation_data, validation_targets, \
test_data, test_targets = data_split(inputs, targets)
        
# Model with 2 inputs, 2 outputs and 3 hidden layers of 25 units
model = frame.Sequential([frame.Linear(2, 25),
                          frame.ReLu(),
                          frame.Linear(25, 25),
                          frame.Tanh(),
                          frame.Linear(25, 25),
                          frame.Sigmoid(),
                          frame.Linear(25, 2),
                          frame.LeakyReLu()])

# Train model, learning rate and number of epochs can be changed for comparison
learning_rate = 0.005
epochs = 100

model, train_error_list, test_error_list, prediction_list = frame.train_validate_model(train_data,
    train_targets, validation_data, validation_targets, model, learning_rate,
    epochs, loss_f = 'MSE')

# Test model on the test data
test_error_values, prediction_list2 = frame.test_model(model, test_data, test_targets)

prediction_list += prediction_list2

# Denormalize data
inputs.mul_(std).add_(mean)
# Plot datapoints and their predicted targets/labels
plt.figure(figsize=(6,6)) 
plt.scatter(inputs[:,0].numpy(), inputs[:,1].numpy(),  c = (np.squeeze(prediction_list)),cmap = 'seismic',alpha=0.7)
plt.title("Predicted dataset in [0,1]², ones are in blue and zeros in red")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()



# Deep Learning framework for proj2
import torch
import math
import time
import datetime

from torch import Tensor

# Parent module for all modules
class Module(object):
    def forward(self, *input):
        raise NotImplementedError
        
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
        
    def param(self):
        return []
    
    
# ReLu activation function
class ReLu(Module):
    def __init__(self):
        super().__init__()
        self.s = 0

    def forward(self, input):
        self.s = input
        relu = input.clamp(min = 0)
        return relu
    
    def backward(self, output_grad):
        input_relu = self.s
        old = input_relu.sign().clamp(min = 0)
        grad = output_grad * old
        return grad    

    def param (self):
        return [(None, None)] 
    
# Leaky ReLu activation function
class LeakyReLu(Module):
    def __init__(self):
        super().__init__()
        self.s = 0

    def forward(self, input):
        self.s = input
        leaky_relu = torch.max(0.1*input,input)
        return leaky_relu
    
    def backward(self, output_grad):
        input_relu = self.s
        old = input_relu.sign().clamp(min = 0) + (-1*input_relu.sign()).clamp(min = 0)*0.1
        grad = output_grad * old
        return grad    

    def param (self):
        return [(None, None)] 
    
# Tanh activation function
class Tanh(Module): 
    def __init__(self):
        super().__init__()
        self.s = 0
        
    def forward(self, input):
        self.s = input   
        # Calculated elementwise and stored as a vector
        tanh_vector = []
        for x in input:
            # We use the formula seen in class instead of 'math.tanh(x)'
            tanh = (2/ (1 + math.exp(-2*x))) -1 
            tanh_vector.append(tanh)

        return torch.FloatTensor(tanh_vector)
    
    def backward(self, output_grad):
        # Derivative formula for the tanh above
        return 4*((self.s.exp() + self.s.mul(-1).exp()).pow(-2)) * output_grad
       
    
    def param (self):
        return [(None, None)]
    
# Sigmoid activation function
class Sigmoid(Module): 
    def __init__(self):
        super().__init__()
        self.s = 0
        
    def forward(self, input):
        self.s = input   

        sigmoid_vector = torch.div(1,(1+torch.exp(-1*torch.FloatTensor(input))))
        
        return sigmoid_vector
    
    def backward(self, output_grad):
        # Derivative formula for the sigmoid above
        sigmoid_vector = torch.div(1,(1+torch.exp(-1*self.s)))
        return sigmoid_vector.mul(1-sigmoid_vector).mul(output_grad)
       
    def param (self):
        return [(None, None)]  
    
# Loss MSE function (forward) and its derivative (backward)
class LossMSE(Module):
    def __init__(self):
        super(LossMSE, self).__init__()
        self.name = 'MSE_loss'
        
    def forward(pred,target):
        #print(pred)
        return (pred - target.float()).pow(2).mean()
    
    def backward(pred,target):
        return 2*(pred - target.float())
    
    def param(self):
        return [(self.w, self.dl_dw), (self.b, self.dl_db)]
    
# Loss MAE function (forward) and its derivative (backward)
class LossMAE(Module):
    def __init__(self):
        super(LossMAE, self).__init__()
        self.name = 'MAE_loss'
        
    def forward(pred,target):
        return (pred-target.float()).abs().mean()
    
    def backward(pred,target):
        return (pred - target.float()).sign().clamp(min = 0).mul(2).add(-1)
    
    def param(self):
        return [(self.w, self.dl_dw), (self.b, self.dl_db)]
    
#SGD optimizer
class SGD():
    def __init__(self, params, learning_rate):
        self.params = params
        self.lr = learning_rate
    
    # Weight update
    def step(self):
        for module in self.params:
            for mod in module:
                if (mod[0] is not None) and (mod[1] is not None):
                    weight, grad = mod
                    weight.add_(-self.lr * grad)
    
    def zero_grad(self):
        # Clear gradients
        for module in self.params:
            for mod in module:  
                if (mod[0] is not None) and (mod[1] is not None):
                    _, grad = mod
                    grad.zero_()

# Fully connected layer module with input and output dimensions         
class Linear(Module):
    def __init__(self, input_dimension, output_dim, eps=1):
        super().__init__()
       
        self.weights = Tensor(output_dim, input_dimension)\
                             .normal_(mean=0, std=eps)
        self.bias = Tensor(output_dim).normal_(0, eps)
        self.x = 0
        self.d_weights = Tensor(self.weights.size())
        self.d_bias = Tensor(self.bias.size())
         
    def forward(self, input):
        self.x = input
        return self.weights.mv(self.x) + self.bias
    
    def backward(self, output_grad):
        self.d_weights.add_(output_grad.view(-1,1).mm(self.x.view(1,-1)))
        self.d_bias.add_(output_grad)
        return self.weights.t().mv(output_grad)
    
    def param(self):
        return [(self.weights, self.d_weights), (self.bias, self.d_bias)]


# Here we make a list of the sequence of modules to use to build the model
# For each module, it executes its forward, backward or param method
class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        args = list(args)[0]

        for ind, module in enumerate(args):
            self.modules.append(module)
    
    def forward(self, input):
        out = input

        for module in self.modules:
            out = module.forward(out)

        return out
    
    def backward(self, output_grad):
        backwards_modules = self.modules[::-1]
        out = output_grad

        for module in backwards_modules:
            out = module.backward(out)
    
    def param (self):
        parameters = []

        for module in self.modules:
            parameters.append(module.param())

        return parameters

# Function that trains the model and uses the validation data as test
def train_validate_model(train_data, train_targets, test_data, test_targets, 
                         model, learning_rate, epochs, loss_f = 'MSE'):      
    # SGD as our optimizer
    sgd = SGD(model.param(), learning_rate)
    
    # init 
    test_error_values = []
    train_error_values = []
    start_time = time.time()
    print('## TRAINING ##')
    
    # For last epoch
    prediction_list=[]
    
    for epoch in range(epochs):
        
        # reset training values each epoch
        training_loss = 0
        nb_train_errors = 0
        nb_valid_errors = 0
        mae = 0

        for n in range(train_data.size(0)):
            # clear gradients
            sgd.zero_grad()
            
            # Use argmax to find the actual target
            train_targets_list = [train_targets[n][0], train_targets[n][1]]
            max_target = train_targets_list.index(max(train_targets_list))
            
            # Output layer
            output = model.forward(train_data[n])
            
            # Get the prediction of the model and its index
            output_list = [output[0], output[1]]
            prediction = output_list.index(max(output_list))
            
            # Increment number of errors if prediction is wrong
            if int(max_target) != int(prediction): 
                nb_train_errors += 1
                
            # If last epoch, keep prediction for plot
            if epoch+1 == epochs:
                prediction_list.append(int(prediction))

            # Chosen loss function
            if loss_f == 'MAE':
                # Compute the accumulative MAE loss and derivative MAE loss
                training_loss += LossMAE.forward(output, train_targets[n])
                d_loss = LossMAE.backward(output, train_targets[n])
            else:
                # Compute the accumulative MSE loss and derivative MSE loss
                training_loss += LossMSE.forward(output, train_targets[n])
                d_loss = LossMSE.backward(output, train_targets[n])

            # Backpropogate loss
            model.backward(d_loss)

            # Update model with a step
            sgd.step()
            
 
        # Store training accuracy for this epoch
        train_acc = (100 * nb_train_errors) / train_data.size(0)   
        train_error_values.append(train_acc)        
        

        # Validation (same as training but with validation data)
        for n in range(test_data.size(0)):
            test_targets_list = [test_targets[n][0], test_targets[n][1]]
            max_target = test_targets_list.index(max(test_targets_list)) 
            
            output = model.forward(test_data[n])
            output_list = [output[0], output[1]]
            prediction = output_list.index(max(output_list))
            if int(max_target) != int(prediction): 
                nb_valid_errors += 1
            
            # If last epoch, keep prediction for plot
            if epoch+1 == epochs:
                prediction_list.append(int(prediction))
        

        training_accuracy = 100-((100 * nb_train_errors) / train_data.size(0))
        validation_accuracy = 100-((100 * nb_valid_errors) / test_data.size(0))

        # Performance check every 25 epochs and the first one
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print('Epoch : {:d}\n \
                  Training loss: {:.02f}\n \
                  Training accuracy: {:.02f}%\n \
                  Validation accuracy {:.02f}%.\n'
              .format(epoch + 1,
                      training_loss,
                      training_accuracy,
                      validation_accuracy))
        
        # store errors
        test_error_values.append((100 * nb_valid_errors) / test_data.size(0))

    # Training time
    end_time = time.time()
    training_time = int(end_time - start_time)
    print("Training/validation time : {:3}\n"
          .format(str(datetime.timedelta(seconds = training_time))))

    return model, train_error_values, test_error_values, prediction_list

# Tests the model in the same fashion as training/validation
def test_model(model, test_data, test_targets):
    #init
    test_error_values = []
    prediction_list=[]
    nb_test_errors = 0
    print('## TESTING ##')
    
    for n in range(0, test_data.size(0)):
        test_targets_list = [test_targets[n][0], test_targets[n][1]]
        true_target = test_targets_list.index(max(test_targets_list))
          
        output = model.forward(test_data[n])
        output_list = [output[0], output[1]]
        prediction = output_list.index(max(output_list))
        if int(true_target) != int(prediction): 
            nb_test_errors += 1
            
        # Keep prediction for plot
        prediction_list.append(int(prediction))

    # Calculate accuracy and print results
    test_accuracy = (100-((100 * nb_test_errors) / test_data.size(0)))
    print('Accuracy on testing set: {:.02f}%'.format(test_accuracy))
    test_error_values.append((100 * nb_test_errors) / test_data.size(0))
    
    return test_error_values, prediction_list
        
    


import torch
from torch import nn
import dlc_practical_prologue as prologue
from models import *


#model parameters (selected by grid search 
lr = 0.005
#loss_ratio is the weight of the auxiliary loss 
loss_ratio = 0.7
mini_batch_size = 100
nb_epochs = 25

#To test every model for 10 trials, this parameter can be changed to 10
nb_trials = 2
model_descriptions= ["Siamese model with auxiliary loss",
                     "Siamese Model without auxiliary loss",
                     "Reduced weight sharing model with auxiliary loss",
                     "Reduced weight sharing model without auxiliary loss",
                     "MLP model" ]

#Models to be tested: 
# 0 : Siamese model with auxiliary loss
# 1 : Siamese Model without auxiliary loss
# 2 : Reduced weight sharing model with auxiliary loss
# 3 : Reduced weight sharing model without auxiliary loss
# 4 : MLP model

print("Models to be tested (in descending order of accuracy) : \n \
                  1- ", model_descriptions[0]," \n \
                  2- ", model_descriptions[1]," \n \
                  3- ", model_descriptions[2]," \n \
                  4- ", model_descriptions[3]," \n \
                  5- ", model_descriptions[4]," \n \n \
By default, this test will run each model for 1 trial, \n \
by changing the nb_trials parameter in the test.py file, models can be run for 10 trials.\n"
      )


#m : to select model
for m in range(1,6): 
    
    print("Testing ",model_descriptions[m-1]," for ",nb_trials," trials: \n")
    train_errs = []
    test_errs = []
    
    # m%2 factor turns the auxiliary loss on or off
    has_aux_loss = bool(loss_ratio*(m%2))
    
    for i in range(nb_trials):
        
        #fixing the seed for each trial to make results repeatable
        torch.manual_seed(i)
        
        #generating the data sets
        train_input, train_target,  train_classes, test_input, test_target, test_classes = \
        prologue.generate_pair_sets(1000)
        
        # normalilizing data  
        train_input.sub_(train_input.mean()).div_(train_input.std())
        test_input.sub_(test_input.mean()).div_(test_input.std())
        
        train_target = train_target.type(torch.FloatTensor)
        test_target = test_target.type(torch.FloatTensor)
        
                
        if m < 3 : #testing siamese 
            
            model = SiameseModel()
            train_model(model, lr, has_aux_loss, train_input, train_target, train_classes, nb_epochs, mini_batch_size, loss_ratio)
            
        elif m < 5  : #testing reduced weight sharing
            
            model = MixedSharingClassifier()
            train_model(model, lr, has_aux_loss, train_input, train_target, train_classes, nb_epochs, mini_batch_size, loss_ratio)
            
        else:  #testing model with only linear layers
            model = MLPClassifier()
            #flattening our data to be used in linear layers
            train_input = torch.flatten(train_input , start_dim=2 )
            test_input = torch.flatten(test_input , start_dim=2)
            
            train_model(model, lr, has_aux_loss, train_input, train_target, train_classes, nb_epochs, mini_batch_size, loss_ratio)
            
        # Calculating training and test error percentages
        
        train_error, tr_class = compute_nb_errors(model, train_input, 
                                                         train_target, train_classes, mini_batch_size )
        test_error, te_class  = compute_nb_errors(model, test_input, 
                                                         test_target, test_classes, mini_batch_size)  
        
        print('training error: {:.02f}% , training auxiliary(class) error {:.02f}% \n\
test error {:.02f}% ,  test auxiliary(class) error {:.02f}% \n\
-----------------------------------------------------------------\n'.format(
                     train_error/ train_input.size(0) * 100,
                    tr_class / train_input.size(0) * 100,
                     test_error / test_input.size(0) * 100,
                    te_class/ test_input.size(0) * 100 ) ) 
    
            
        train_errs.append(train_error/ train_input.size(0) * 100)
        test_errs.append(test_error / test_input.size(0) * 100)
      
    #gives perforamnce estimate if models are run for multiple rounds
    if nb_trials > 1:
        
        train_errs = torch.tensor(train_errs)
        test_errs = torch.tensor(test_errs)
        
        print('Performance estimate for {} trials :\n\
train_error: {:.03f}% , test_error: {:.03f}% , std of test err: {:.03f}'.format(
                    nb_trials,
                    train_errs.mean(),
                    test_errs.mean(),    
                    test_errs.std() )  )

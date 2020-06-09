import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

'''digit classifier block for Siamese model with conv2d based feautre extracter + linear layers '''
class DigitClassifierBlock(nn.Module):

    def __init__(self, nb_hidden=64):
        super( DigitClassifierBlock, self).__init__()
        
        #feature extractor conv layers
        
        # cl1: Nx1x14x14 ->  Nx32x14x14
        self.cl1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        # cl2:  Nx32x14x14 ->  Nx32x14x14
        self.cl2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  
        
        #digit discrimination linear layers
        
        self.fc1 = nn.Linear(32*7*7, 64) 
        #dropout to reduce overfitting, parameter p has been optimized
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, nb_hidden)
        self.fc3 = nn.Linear(nb_hidden, 10)
        
    def forward(self, x):
        
        x = F.relu(self.cl1(x))
        x = F.max_pool2d(F.relu(self.cl2(x)), kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(-1, 32*7*7)))
        x = self.dropout(x)  #to reduce overfitting
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        
        return x
    
#-----------------------------------------------------------------------------------    
'''digit classifier block for Siamese model with flattened input and only linear layers 
   the layers are chosen to imitate the original layer sizes of the DigitClasifierBlock'''
class MLPDigitBlock(nn.Module):

    def __init__(self, nb_hidden=64):
        super( MLPDigitBlock, self).__init__()
        
        
        self.fc1 = nn.Linear(1*14*14, 32*14*14)
        #maxpool2D of the DigitClassifierBlock is replaced with a simple size change
        self.fc2 = nn.Linear(32*14*14, 32*7*7) 
        
        self.fc3 = nn.Linear(32*7*7, 64)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(64, nb_hidden)
        self.fc5 = nn.Linear(nb_hidden, 10)
        
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)) )
        x = F.relu(self.fc3(x))
        x = self.dropout(x)  #to reduce overfitting
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        
        
        return x    
    
    
#-----------------------------------------------------------------------------------    
class SiameseModel(nn.Module):
     
    def __init__(self, nb_hidden=64):
        super( SiameseModel, self).__init__()
        #digitClassifierBlock: Nx1x14x14 -> Nx2x10
        self.digitBlock = DigitClassifierBlock(nb_hidden) 
        self.fc4 = nn.Linear(20 , 16)
        #since our problem is a binary classification problem, one output with 0/1 value is sufficient
        #alternatively, one-hot-encoded output could be used and same results would be obtained
        self.fc5 = nn.Linear(16 , 1)

    def forward(self, x):
        # inputting each digit seperately to the digit classifier
        digit1 = self.digitBlock( x[:,:1,:,:] )
        digit2 = self.digitBlock( x[:,1:,:,:] )

        # Merging the outputs of the digit block for the two digits
        x = torch.cat( (digit1.clone(),digit2.clone()) ,1)
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x)).view(-1)
        return x, digit1, digit2
    
#-----------------------------------------------------------------------------------
'''version of the siamese model with only linear layers'''
class MLPClassifier(nn.Module):
     
    def __init__(self, nb_hidden=64):
        super( MLPClassifier, self).__init__()
        
        self.digitBlock = MLPDigitBlock(nb_hidden)
        self.fc6 = nn.Linear(20 , 16)
        self.fc7 = nn.Linear(16 , 1)
    
    def forward(self, x):
        # Divide in 2 datasets, one for each channel
        digit1 = self.digitBlock( x[:,0,:])
        digit2 = self.digitBlock( x[:,1,:])

        # Concatenate 
        x = torch.cat( (digit1.clone(),digit2.clone()) ,1)
        x = F.relu(self.fc6(x))
        x = torch.sigmoid(self.fc7(x)).view(-1)
        return x, digit1, digit2
    
    
#-----------------------------------------------------------------------------------
''' version of the siamese model with some layers of the digit classifier no longer sharing weights, no hlper digit block used '''
class MixedSharingClassifier(nn.Module):
     
    def __init__(self, nb_hidden=64):
        super( MixedSharingClassifier, self).__init__()
        
        
        self.cl1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        #non-weight-sharing layer
        self.cl2a = nn.Conv2d(32, 32, kernel_size=3, padding=1)  
        self.cl2b = nn.Conv2d(32, 32, kernel_size=3, padding=1)  
        
        self.fc1 = nn.Linear(32*7*7, 64)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        #non-weight-sharing layer
        self.fc2a = nn.Linear(64, nb_hidden)
        self.fc2b = nn.Linear(64, nb_hidden)
        self.fc3 = nn.Linear(nb_hidden, 10)
               
        self.fc4 = nn.Linear(20 , 16)
        self.fc5 = nn.Linear(16 , 1)
        self.dropout2 = torch.nn.Dropout(p=0.5)
     
        
    def forward(self, x):
        # Divide in 2 datasets, one for each channel
        digit1 = x[:,:1,:,:] 
        digit2 = x[:,1:,:,:] 
            
        digit1 = F.relu(self.cl1(digit1))
        digit2 = F.relu(self.cl1(digit2))
        #non-weight-sharing layer
        digit1 = F.max_pool2d(F.relu(self.cl2a(digit1)), kernel_size=2, stride=2)
        digit2 = F.max_pool2d(F.relu(self.cl2b(digit2)), kernel_size=2, stride=2)
        digit1  = F.relu(self.fc1(digit1.view(-1, 32*7*7)))
        digit2  = F.relu(self.fc1(digit2.view(-1, 32*7*7)))
        digit1 = self.dropout1(digit1)  #to reduce overfitting
        digit2 = self.dropout1(digit2)  #to reduce overfitting
        #non-weight-sharing layer
        digit1 = F.relu(self.fc2a( digit1))
        digit2 = F.relu(self.fc2b( digit2))
        digit1  = self.fc3(digit1 )
        digit2 = self.fc3(digit2)
        
        # Concatenate
        x = torch.cat( (digit1.clone(),digit2.clone()) ,1)
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x)).view(-1)
        return x, digit1, digit2       
    
    
#-----------------------------------------------------------------------------------    
# error computation and training methods are common for all models
    

def compute_nb_errors(model, test_input, test_target, test_classes, mini_batch_size):
     model.train(mode=False)
     nb_errors = 0
     class_errors = 0
     for b in range(0, test_input.size(0), mini_batch_size):
        output = model(test_input.narrow(0, b, mini_batch_size))
        predictions,classes1,classes2 = output
        predictions.round_()
        
        _, classes1 = classes1.max(1)
        _, classes2 = classes2.max(1)
        
        
        for k in range(mini_batch_size):
           #final result error
           if test_target[b + k] != predictions[k]:
               nb_errors = nb_errors + 1
           #digit classification error
           if test_classes[b + k,0] != classes1[k]:
                class_errors = class_errors + 1
           if test_classes[b + k,1] != classes2[k]:
                class_errors = class_errors + 1

     return nb_errors, class_errors
   
#----------------------------------------------------------------------------------- 
     
def train_model(model, lr, has_aux_loss, train_input, train_target, train_classes, nb_epochs, mini_batch_size, loss_ratio ):
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    #criterion for the final decision 0/1
    #the sigmoid layer at the end of the forward pass + BCELoss -> gives same results as CrossEntropyLoss with one-hot-encoded
    criterion1 = nn.BCELoss()
    #criterion for digit classification result
    criterion2 = nn.CrossEntropyLoss()
    digit1_classes = train_classes[:,0]
    digit2_classes = train_classes[:,1]

    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            
            output, aux_out1, aux_out2 = model(train_input.narrow(0, b, mini_batch_size))
            loss1 = criterion1(output, train_target.narrow(0, b, mini_batch_size))

            loss2 = criterion2(aux_out1, digit1_classes.narrow(0, b, mini_batch_size))
            loss3 = criterion2(aux_out2, digit2_classes.narrow(0, b, mini_batch_size))
                   
            if has_aux_loss:  loss = loss1 + loss_ratio * (loss2 +  loss3 )
            else: loss = loss1
            
            model.zero_grad()
            loss.backward()
            sum_loss = sum_loss + loss.item()
            optimizer.step()
            
        #print(e, sum_loss)


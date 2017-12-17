import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))                               
        self.lr = learning_rate
        
        self.activation_function = lambda x : 1/(1+np.exp(-x))  # Replace 0 with your sigmoid calculation.
    
        self.sigmoidf = lambda x: self.activation_function(x)*(1-self.activation_function(x))

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs,
                                                                        X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch
            X incomes as an (51,) array
        '''
        #### Implement the forward pass here ####
        
        inputs = np.array(X, ndmin=2) #array([1,2,3,4,5]) -> array([[1,2,3,4,5]])
                
        hidden_inputs = np.dot(inputs, self.weights_input_to_hidden) 
        '''[x1,x2,x3,x4,x5,Xn]*[w11 ,w21] = array([[h1,hn]])
                               [w21 ,w22]   
                               [wij ,wij]
            hidden_outputs sera uma matriz nx1, sendo n o numero de hidden units.
        '''   
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) 
        '''
        [h1,h2] *[w1] = h1*w1+h2*w2 = scalar
                 [w2]
        
        the derivative of the f(x) = x, df(x)/dx = 1, so the final output will be 1*final_input
        '''
        
        final_outputs = final_inputs
        
        return final_outputs, hidden_outputs


    def backpropagation(self, final_outputs,hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
        '''
        #### Implement the backward pass here ####
        ### Backward pass ###
        new_X = np.array(X, ndmin=2)
        #new_X = new_X.reshape((1, -1))
        new_y = np.array(y, ndmin=2)
        #new_y = new_y.reshape((1, -1))

        # TODO: Output error - Replace this value with your calculations.
        error = final_outputs - new_y # Output layer error is the difference between desired target and actual output.
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term =  error

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_output_error_term = np.dot(output_error_term, self.weights_hidden_to_output.T)
        hidden_input_error_term = hidden_output_error_term * hidden_outputs * (1 - hidden_outputs)

        # Weight step (input to hidden)
        delta_weights_i_h += np.dot(new_X.T, hidden_input_error_term)
        # Weight step (hidden to output)
        delta_weights_h_o += np.dot(hidden_outputs.T, output_error_term)
        return delta_weights_i_h, delta_weights_h_o
    
    
    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records
        '''
        self.weights_hidden_to_output += -self.lr*delta_weights_h_o / n_records
        self.weights_input_to_hidden += -self.lr*delta_weights_i_h / n_records 

    def run(self, features):
        # Forward pass 
        ### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs# signals from final output layer 
        
        return final_outputs

#########################################################
# Set your hyperparameters here
##########################################################
iterations = [500,1000,2000]
learning_rate = 0.1
hidden_nodes = [5,10,15,20,25,50]
output_nodes = 1

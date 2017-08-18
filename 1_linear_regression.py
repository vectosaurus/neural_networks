import tensorflow as tf 
import pandas as pd
import numpy as np 
from utils import activation_functions, optimizers, regressor_accuracy_metrics
from helperfins import batch_indices

# Outline:
# The code will be divided in three parts 
    # - the zeroth part will be where I import the libraries and define all the helper 
    # functions
    # - first will be the estimator definition,
    # wherein I'll be writing a regressor class which performs all the calculations 
    # required for training the model 
    # - and the second part will contain an example 
    # outlining the usage of the features of the regressor class.

# Estimator
# Creating an instance of the LinearRegressor class takes no inputs
# The methods available with the LinearRegressor are. -
#   fit: takes in the following parameters
#       X: predictor variables, covariates
#       Y: target values, response variable
#       epochs: number of epochs for which the model will be trained
#       batch_size: model parameters will be updated after considering the 
#                   cumulative loss from `batch_size` number of examples
#       learning_rate: the rate at which cost gradients will be subtracted from the 
#                      parameters
#       optimizer: select the optimizer 
#       train_fraction(excluded): fraction of the input data `X` and `Y` that should be used.
#                       the remainder of the data can be used as validation data
#       display_rate: If train_fraction < 1, the test cost will be displayed after
#                     every `display_rate` number of epochs

class LinearRegressor(object):
    def __init__(self):
        
    def fit(self,X,y,epochs,batch_size,learning_rate,optimizer,train_fraction,\
            display_rate):
        """Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples, n_targets]
            Target values
        epochs: integer. number of training cycles. default value = 10
        batch_size: integer. training data will be split into batches with `batch_size` 
                    examples in one batch (the last batch may be smaller)
                    default value = 100
        learning_rate: float. the learning rate that will be provided to the optimizer
                        default value = 0.01
        optimizer: string. the acceptable strings are 
                    # AdadeltaOptimizer
                    # AdagradOptimizer
                    # AdagradDAOptimizer
                    # MomentumOptimizer
                    # AdamOptimizer
                    # FtrlOptimizer
                    # ProximalGradientDescentOptimizer
                    # ProximalAdagradOptimizer
                    # RMSPropOptimizer
        display_rate: integer. the total training cost will be displayed at an interval 
                      of `display_rate` epochs        
        """
        
        # set the attributes
        print('Beginning model training...')
        self.X = X
        self.y = y
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_fn = optimizers[optimizer]
        self.display_rate = display_rate
        
        # data parameters, needed for weight and bias initialization
        n_samples = X.shape[0]
        input_features = X.shape[1]
        output_features = y.shape[1]
        
        batch_ind = batch_indices(n_samples,self.batch_size)
        num_batches = len(batch_ind)-1
        
        # weight and bias initialization
        print('Intializing model parameters...')
        self.W = tf.Variable(np.random.normal(loc=0.0,
                                              scale=1/(input_features**-0.5),
                                              size=(input_features,output_features)),
        trainable=True, name = 'train_weights', dtype = 'float64')
        self.b = tf.Variable(np.zeros((output_features)),
        trainable=True, name = 'train_weights', dtype = 'float64')
        
        # define the batch x, y, y_hat
        self.xt = tf.placeholder('float64',(None,n_features), name = 'x')
        self.yt = tf.placeholder('float64',(None,n_features), name = 'y_true')
        self.yp = tf.add(tf.matmul(self.xt,self.W), self.b)
        
        # squared loss. Loss type can also be a type of type of estimator 
        self.cost = tf.reduce_mean(tf.square(y_train_batch-y_hat),reduction_indices=1)
        self.optimizer = self.optimizer_fn(self.learning_rate).minimize(cost)
        
        # compute the graph
        _ = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(_)
        
        # to store costs
        self.costs = []
        
        print('Beginning model training...')
        for epoch in range(self.epochs):
            for batch_num in range(num_batches-1):
                x_batch = X[batch_ind[batch_num]:batch_ind[batch_num+1]]
                y_batch = y[batch_ind[batch_num]:batch_ind[batch_num+1]]
                self.sess.run(optimizer,
                              feed_dict={self.xt : x_batch,self.yt : y_batch})
            cost = sess.run(self.cost,
                              feed_dict={self.xt : X, self.yt : y})
            self.costs.append(cost)
            if epoch%self.display_rate == 0:
                print('Epoch: \t\t', epoch,'Cosr: \t\t', round(self.costs[-1],2))
        print('Model training complete.')
        
    def predict(self,X):
        if not X:
            pred = self.sess.run(y_hat,feed_dict={self.xt=self.X})
            return pred
        return self.sess.run(y_hat,feed_dict={self.xt=X})
    
    def save(self,filename):  
        try:  
            if not filename:
                filename = 'MultiLayerPerceptron_ModelNo_'+str(self.save_index)+'.chkp'
            saver = tf.train.Saver()
            saver.save(sess, filename)
            self.save_index+=1
            print("Model saved: "+filename)
        except:
            print('Failed to save model!')
    
    def update(self,X,Y):
        self.X = np.concatenate((self.X,X),axis=0)
        self.Y = np.concatenate((self.Y,Y),axis=0)
        n_samples = X.shape[0]
        batch_ind = batch_indices(n_samples,self.batch_size)
        print('Updating current model...')
        batch_ind.sort()
        num_batches = len(batch_ind)-1
        
        for epoch in range(self.epochs):
            for batch_num in range(num_batches):
                x_batch = X[batch_ind[batch_num]:batch_ind[batch_num+1]]
                y_batch = y[batch_ind[batch_num]:batch_ind[batch_num+1]]
                self.sess.run(self.optimizer, feed_dict={self.xt:x_batch, self.yt:y_batch})
            cost = self.sess.run(self.cost, feed_dict={self.xt:self.X, self.yt:self.Y})
            self.costs.append(cost)
            
            if (epoch)%self.display_rate == 0:
                print('\tEpoch:', epoch, 'Training Cost: ', self.costs[-1])
        
        
# example
n_input = [784]
n_hidden = [400]
n_output = [10]
n_samples = 10000
layer_dimensions = n_input+n_hidden+n_output
X = np.asarray(np.random.normal(size = (n_samples,n_input[0])))
true_weights,bias = tr_weight_init(layer_dimensions) #[i.shape for i in true_weights]
Y = tr_reducer(X,true_weights,bias,identity) # Y.sum(axis=1).sum() == 10000
uX = np.asarray(np.random.normal(size = (n_samples//3,n_input[0])))
uY = tr_reducer(uX,true_weights,bias,identity) # Y.sum(axis=1).sum() == 10000

# parameters
train_fraction = 0.7
epochs = 100
batch_size = 100
activation_function = 'softmax'
optimizer = 'AdagradOptimizer'
learning_rate = 0.001
normalization = False
dropout = 0.0

# run the damn thing.
mlp = MultiLayerPerceptron(layer_dimensions)
mlp.fit(X,Y,train_fraction,epochs,batch_size,activation_function,\
            optimizer,learning_rate,normalization,dropout)
y_pred = mlp.predict(X)
acc1 = mlp.accuracy(Y,metric=None)
print(acc1)
mlp.update(uX,uY)
acc2 = mlp.accuracy(Y,metric=None)
print(acc2)

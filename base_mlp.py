import tensorflow as tf
import numpy as np
from functools import reduce
from sklearn.metrics import r2_score,explained_variance_score,mean_absolute_error\
,mean_squared_error,mean_squared_log_error,median_absolute_error,r2_score

# outline
# class mlp
#   input_params: activation function, layer dimensions(i,h,o), dropout, optimizer, learning_rate, epochs
#   train_params: activation function, optimizer, learning_rate, epochs, batch_size
#   sklearnesque: predict, fit
#   methods = fit(creates new_model, accuracy_metric,predicts on new_data(optional)), update(model), save(filename), predict(X), 
#   need to select cost type as well: cross-entropy, squared loss, etc

activation_functions_dict = {'relu': tf.nn.relu,
  'tanh': tf.nn.tanh,
  'sigmoid': tf.nn.sigmoid,
  'elu': tf.nn.elu,
  'softplus': tf.nn.softplus,
  'softsign': tf.nn.softsign,
  'relu6': tf.nn.relu6,
  'softmax':tf.nn.softmax,
  None: lambda x:x
}

optimizers_dict = {'GradientDescentOptimizer': tf.train.GradientDescentOptimizer,
    'AdadeltaOptimizer': tf.train.AdadeltaOptimizer, 
    'AdagradOptimizer': tf.train.AdagradOptimizer, 
    'AdagradDAOptimizer': tf.train.AdagradDAOptimizer, 
    'MomentumOptimizer': tf.train.MomentumOptimizer, 
    'AdamOptimizer': tf.train.AdamOptimizer, 
    'FtrlOptimizer': tf.train.FtrlOptimizer, 
    'ProximalGradientDescentOptimizer': tf.train.ProximalGradientDescentOptimizer, 
    'ProximalAdagradOptimizer': tf.train.ProximalAdagradOptimizer, 
    'RMSPropOptimizer': tf.train.RMSPropOptimizer,
    None: tf.train.GradientDescentOptimizer
}

accuracy_metrics_dict = {'r2_score':r2_score
,'explained_variance_score':explained_variance_score
,'mean_absolute_error':mean_absolute_error
,'mean_squared_error':mean_squared_log_error
,'mean_squared_log_error':mean_squared_log_error
,'median_absolute_error':median_absolute_error
,None:r2_score
}

def identity(X):
    return X

def softmax(X):
    """Compute softmax values for each sets of scores in x."""
    e_X = np.exp(X - np.max(X, axis = 1).reshape(-1,1))
    res = e_X/np.sum(e_X, axis=1).reshape(-1,1)
    return res

def one_hot(Y,axis=1):
    arg_max = np.argmax(Y, axis=axis).reshape(1,-1)
    ohe = np.zeros((Y.shape[0], Y.shape[1]))
    ohe[np.arange(Y.shape[0]), arg_max] = 1
    return ohe

def reducer(x,W,b,activation_fn):
    for i in range(len(W)):
        x = activation_fn(tf.add(tf.matmul(x,W[i]),b[i]))
    return x

def tr_reducer(x,W,b,activation_fn):
    for i in range(len(W)):
        x = activation_fn(np.matmul(x,W[i])+b[i])
    return x

def weight_init(layer_dimensions):
    """
    Returns initialized weights and biases for a feed forward network.
    Input: array-like, list, tuple. An iterable.
    Initializes weight using a normal distribution with mean zero and standard deviation equal to the square root of the
    incoming connections to a node.
    
    Example:
    layer_dimensions = [3,6,9]
    weight_init(layer_dimensions)
    """
    W = [None]*(len(layer_dimensions)-1)
    b = [None]*(len(layer_dimensions)-1)
    for i in range(len(layer_dimensions)-1):
        W[i] = tf.Variable(np.random.normal(loc=0.0, scale=np.sqrt(layer_dimensions[i]), size=(layer_dimensions[i],layer_dimensions[i+1])),\
                        trainable=True, name=str('model_weights_'+str(i)+'_'+str(i+1)), dtype='float64')
        b[i] = tf.Variable(np.zeros(shape=(layer_dimensions[i+1])), trainable=True, name=str('model_bias_'+str(i)+'_'+str(i+1)), dtype='float64')
    return W,b

def tr_weight_init(layer_dimensions):
    W = [None]*(len(layer_dimensions)-1)
    b = [None]*(len(layer_dimensions)-1)
    for i in range(len(layer_dimensions)-1):
        W[i] = np.random.normal(loc=0.0, scale=np.sqrt(layer_dimensions[i]), size=(layer_dimensions[i],layer_dimensions[i+1]))
        b[i] = np.zeros(shape=(layer_dimensions[i+1]))
    return W,b

class MultiLayerPerceptron(object):
    def __init__(self, layer_dimensions):
        self.layer_dimensions = layer_dimensions
        self.W, self.b = weight_init(layer_dimensions)
        self.num_layers = len(layer_dimensions) # includes input and output layer
    
    def fit(self,X,Y,train_fraction=1,epochs=10,batch_size=64,activation_function=None,\
            optimizer=None,learning_rate=0.01,normalization=False,dropout=0.0,display_in=5):
        """
        The fit function takes in multiple inputs (discussed below) and trains a feed forward network
        Input Parameters
            X: the predictor variables
            Y: the target variables
            epochs: defaults to 10. Number of training cycles
            batch_size: defaults to 64. Model will update weights using batch_size number of examples at once
            activation_function: default None. activation function to be used to introduce non-linearity. 
                Accepted values are None, relu, sigmoid, tanh.
            optimizer: defaults to Gradient Descent. optimizer to be used to train the model
            learning_rate: defaults to 0.01. In case of adaptive learning rate optimizers, the learning rate 
                is the starting point
            normalization: boolean, defaults to False. Whether the input variables should be normalised. 
                The entire dataset will be normalised. In case only select variables are to be normalized, preprocess the
                data and then fir.
        """
        # set class attributes
        print('Setting up model parameters...')
        self.X = X
        self.Y = Y
        self.train_fraction = train_fraction
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation_function = activation_functions_dict[activation_function] # stage II, currently using None
        self.optimizer_function = optimizers_dict[optimizer] # stage II, using GradientDescentOptimizer
        self.learning_rate = learning_rate
        self.normalization = normalization
        # self.dropout = tf.nn.dropout(0) # for now
        self.costs_train = [] # store total cost of train examples at end of every iteration
        self.costs_test = [] # store total cost of test examples at end of every iteration
        self.display_rate = display_in
        self.save_index = 0
        
        # split data in test and train
        print('Data preprocessing...')
        train_size = int(self.X.shape[0]*train_fraction)
        test_size = self.X.shape[0] - train_size
        train_ind = np.full(shape=(self.X.shape[0],), fill_value = False)
        train_ind[np.random.choice(np.arange(0,self.X.shape[0]), size = train_size, replace = False)] = True
        train_X, test_X = self.X[train_ind,:], self.X[np.logical_not(train_ind),:]
        train_Y, test_Y = self.Y[train_ind,:], self.Y[np.logical_not(train_ind),:]
        batch_ind = list(set(list(range(0,train_size,self.batch_size))+[n_samples-1]))
        batch_ind.sort()
        num_batches = len(batch_ind)-1
        
        # building the graph
        self.x = tf.placeholder('float64', (None,self.layer_dimensions[0]), name = 'train_x_batch')
        self.y = tf.placeholder('float64', (None,self.layer_dimensions[-1]), name = 'train_y_batch')
        self.y_hat = reducer(self.x, self.W, self.b, self.activation_function)
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.y_hat), reduction_indices=1))
        self.optimizer = self.optimizer_function(self.learning_rate).minimize(self.cost)
        
        # computing the graph
        print('Training the model...')
        _ = tf.global_variables_initializer()
        self.sess= tf.Session()
        self.sess.run(_)
        
        for epoch in range(self.epochs):
            for batch_num in range(num_batches):
                x_batch = train_X[batch_ind[batch_num]:batch_ind[batch_num+1]]
                y_batch = train_Y[batch_ind[batch_num]:batch_ind[batch_num+1]]
                self.sess.run(self.optimizer, feed_dict={self.x:x_batch, self.y:y_batch})
            cost_train = self.sess.run(self.cost, feed_dict={self.x:train_X, self.y:train_Y})
            cost_test = self.sess.run(self.cost, feed_dict={self.x:test_X, self.y:test_Y})
            self.costs_train.append(cost_train)
            self.costs_test.append(cost_test)
            
            if (epoch)%self.display_rate == 0:
                print('\tEpoch:', epoch, 'Test Cost: ', self.costs_train[-1])
        print('Finished model training.')
    
    def predict_(self,X):
        return self.sess.run(self.y_hat, feed_dict={self.x:X})
    
    def predict(self,X=None):
        # if isinstance(X, None):
        #     return self.predict_(X)
        # else:
        #     return self.predict_(self.X)
        y_hat = self.predict_(self.X)
        # pred_ = self.sess.run(tf.argmax(y_hat, 1))
        return y_hat
        
    def accuracy(self,X=None,Y=None,metric=None):
        if Y:
            pred_ = self.predict(X)
            return accuracy_metrics_dict[metric](Y,pred_)
        else:
            pred_ = pred_ = self.predict(self.X)
            return accuracy_metrics_dict[metric](self.Y,pred_)
    
    def save(self, filename):  
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
        batch_ind = list(set(list(range(0,n_samples,self.batch_size))+[n_samples-1]))
        print('Updating current model...')
        batch_ind.sort()
        num_batches = len(batch_ind)-1
        
        for epoch in range(self.epochs):
            for batch_num in range(num_batches):
                x_batch = X[batch_ind[batch_num]:batch_ind[batch_num+1]]
                y_batch = Y[batch_ind[batch_num]:batch_ind[batch_num+1]]
                self.sess.run(self.optimizer, feed_dict={self.x:x_batch, self.y:y_batch})
            cost_train = self.sess.run(self.cost, feed_dict={self.x:self.X, self.y:self.Y})
            self.costs_train.append(cost_train)
            
            if (epoch)%self.display_rate == 0:
                print('\tEpoch:', epoch, 'Training Cost: ', self.costs_train[-1])

# Example
#create training data
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



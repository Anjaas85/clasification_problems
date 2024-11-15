import numpy as np
from libreria.math import sigmoid

class LogisticRegression:
    def __init__(self, num_features : int):
        self.parameters = np.random.normal(0, 0.01, num_features)
        
    def predict(self, x:np.array) -> np.array:
        """
        Method to compute the predictions for the input features.

        Args:
            x: it's the input data matrix.

        Returns:
            preds: the predictions of the input features.
        """
        ##############################
        ###     YOUR CODE HERE     ###
        ##############################
        
        preds = sigmoid(np.dot(x, self.parameters))
        return preds
    
    @staticmethod
    def likelihood(preds, y : np.array) -> np.array:
        """
        Function to compute the log likelihood of the model parameters according to data x and label y.

        Args:
            preds: the predicted labels.
            y: the label array.

        Returns:
            log_l: the log likelihood of the model parameters according to data x and label y.
        """
        ##############################
        ###     YOUR CODE HERE     ###
        ##############################

        # compute the log likelihood using the defined formula
        log_l = np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))
        return log_l
    
    def update_theta(self, gradient: np.array, lr : float = 0.5):
        """
        Function to update the weights in-place.

        Args:
            gradient: the gradient of the log likelihood.
            lr: the learning rate.

        Returns:
            None
        """
        ##############################
        ###     YOUR CODE HERE     ###
        ##############################

        # update the parameters using the gradient
        self.parameters = self.parameters + lr * gradient
        
    @staticmethod
    def compute_gradient(x : np.array, y: np.array, preds: np.array) -> np.array:
        """
        Function to compute the gradient of the log likelihood.

        Args:
            x: it's the input data matrix.
            y: the label array.
            preds: the predictions of the input features.

        Returns:
            gradient: the gradient of the log likelihood.
        """
        ##############################
        ###     YOUR CODE HERE     ###
        ##############################

        # x has shape (N, D) and y has shape (N,)
        # so take transpost of x and multiply with the difference of y and preds, shape (D, N) * (N,) = (D,)
        # then divide by the number of samples to get the average gradient
        gradient = np.dot(x.T, (y - preds)) / len(y)

        return gradient


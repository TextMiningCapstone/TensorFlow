##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
	c=sqrt(6/(m+n))
	A0=random.uniform(-c,c,(m,n))

    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0

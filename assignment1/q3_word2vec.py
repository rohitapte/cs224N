#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    x=x/(np.linalg.norm(x,axis=1,keepdims=True))
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    #y_hat=p(o|c)=softmax
    #here output vectors are as rows so we just need U v_c instead of U^T v_C
    #CE=-sum ylog(y_hat)
    #dCE/dv_c=U(y_hat-y)
    #dCE/du_k=v_c(y_hat-y)
    y_hat=softmax(outputVectors.dot(predicted))
    #print(y_hat.shape)
    #since y is one hot vector we only need cost for element i
    cost=-np.log(y_hat[target])
    #reshape y_hat and create one hot vector for y
    y=np.zeros((y_hat.shape[0],1))
    y[target]=1
    y_hat=y_hat.reshape((y_hat.shape[0],1))
    y_hat_minus_y=y_hat-y
    #print(y_hat_minus_y)
    #print(y_hat_minus_y.shape)
    #print(outputVectors)
    #print(outputVectors.shape)
    #here output vectors are as rows so instead of U(y_hat-y) we need to do U^T (y_hat-y)
    #gradPred=dCE/d_v_c
    gradPred=outputVectors.T.dot(y_hat_minus_y)
    #print(gradPred)
    predicted_reshaped=predicted.reshape((1,predicted.shape[0]))
    grad=y_hat_minus_y.dot(predicted_reshaped)
    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    cost=0.0
    grad=np.zeros((outputVectors.shape))
    #same as above, output vectors are as rows so we just need U instead of U^T
    #cost=-log(sigma(u_o vc))- sum_k=1^K log (sigma(-u_k vc))
    sigmoid_o=sigmoid(outputVectors[target].dot(predicted))
    cost+=-np.log(sigmoid_o)
    gradPred=(sigmoid_o-1)*outputVectors[target]
    grad[target]=(sigmoid_o-1)*predicted
    #start from the first index since 0 is target
    for i,item in enumerate(indices[1:]):
        sigmoid_k_i=sigmoid(-outputVectors[item].dot(predicted))
        cost+=-np.log(sigmoid_k_i)
        gradPred-=(sigmoid_k_i-1)*outputVectors[item]
        grad[item]+=-(sigmoid_k_i-1)*predicted
    gradPred=gradPred.reshape((gradPred.shape[0],1))
    #print(cost)
    #print(grad)
    #print(gradPred)
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    centerWordLocation=tokens[currentWord]
    v_hat=inputVectors[centerWordLocation]
    #as per lecture, go thro each position t around center words t
    #and calculate p(o|c)
    #total cost is the sum
    cost=0.0
    for word in contextWords:
        wordLocation=tokens[word]
        cost_fxn, gradPred_fxn, grad_fxn=word2vecCostAndGradient(v_hat,wordLocation,outputVectors,dataset)
        cost+=cost_fxn
        gradOut+=grad_fxn
        #print(gradOut)
        #print(gradIn)
        #print(gradPred_softmax)
        #print(gradIn.shape)
        #print(gradPred_softmax.shape)
        gradPred_fxn=gradPred_fxn.reshape((gradPred_fxn.shape[0]))
        #print(gradPred_softmax.shape)
        #print(gradPred_softmax)
        gradIn[centerWordLocation,:]+=gradPred_fxn
        #print(gradIn.shape)
        #print(gradPred_softmax.shape)
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    #CBOW is just reversed skip gram. so the probability is the sum of the probability of the context words
    nearbyWordIndex=[tokens[i] for i in contextWords]
    combinedProb=np.sum(inputVectors[nearbyWordIndex],axis=0)
    #our target word is the center word
    centerWordLocation=tokens[currentWord]
    #no need for loop here. just one call since we are only calculating the center prob
    cost_fxn, gradPred_fxn, grad_fxn = word2vecCostAndGradient(combinedProb,centerWordLocation,outputVectors,dataset)
    cost+=cost_fxn
    gradOut+=grad_fxn
    gradPred_fxn=gradPred_fxn.reshape((gradPred_fxn.shape[0]))
    #loop thro the nearbywords to update gradIn
    for i in nearbyWordIndex:
        gradIn[i,:]+=gradPred_fxn
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()

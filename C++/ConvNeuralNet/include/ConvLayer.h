//
// Created by agils on 6/10/2022.
//

#pragma once

#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <memory.h>
#include <iomanip>
#include <random>
#include <chrono>

#define matrix vector<vector<double>>
#define tensor vector<matrix>
#define intTensor vector<vector<vector<int>>>

using std::vector;
using std::uniform_real_distribution;
using std::default_random_engine;
using std::max;
using std::exception;
using std::ifstream;
using std::ios;
using std::getline;
using std::stod;

/**
 * Element-wise multiplication of two tensors
 *
 * @param t1 Tensor to be multiplied by t2
 * @param t2 Multiplies t1
 */
void multTensors(tensor& t1, tensor& t2);
/**
 * Flattens a tensor into a vector
 *
 * @param dataIn input tensor
 * @return vector containing elements of dataIn
 */
vector<double> flattenTensor(tensor dataIn);
/**
 * Reshapes a vector into a tensor (ex: flattened tensor back to tensor)
 *
 * @param vec input vector
 * @param channelCount number of channels in tensor
 * @param r number of rows in each channel
 * @param c number of columns in each channel
 * @return tensor containing the reshaped data form vec
 */
tensor reshapeVector(vector<double> vec, int channelCount, int r, int c);
/**
 * Adds padding to the edges of a tensor
 *
 * @param dataIn tensor to be padded
 * @param padWidth width of padding
 * @return dataIn with padding
 */
tensor addPadding(tensor dataIn, int padWidth);
/**
 * Removes the padding on the edges of a tensor
 *
 * @param dataIn padded tensor
 * @param padWidth width of padding
 * @return dataIn with padding removed
 */
tensor delPadding(tensor dataIn, int padWidth);
/**
 * Creates an r x c matrix with default values
 *
 * @param r rows in the matrix
 * @param c columns in the matrix
 * @param val default value of the matrix
 * @return r x c matrix
 */
matrix allocMatrix(int r, int c, double val=0.0);
/**
 * Creates channelCount x dim x dim tensor with default values
 *
 * @param channelCount number of channels in the tensor
 * @param dim dimension of each channel
 * @param value default value of the tensor
 * @return channelCount x dim x dim tensor
 */
tensor allocTensor(int channelCount, int dim, double value=0.0);
/**
 * Creates a size (numTensors) vector of channelCount x dim x dim tensors
 *
 * @param numTensors number of tensors in the vector
 * @param channelCount number of channels in each tensor
 * @param dim dimension of each channel
 * @param value default value of the tensor vector
 * @return vector containing (numTensors) channelCount x dim x dim tensors
 */
vector<tensor> allocTensVec(int numTensors, int channelCount, int dim, double value=0.0);
/**
 * Given a value range, applies min-max normalization to each tensor in a vector
 *
 * @param dataIn vector of input tensors (images)
 * @param minMaxRanges vector of minimum and maximum values of the tensors
 */
void normalize(vector<tensor>& dataIn, vector<double> minMaxRanges);

/**
 * Dense Layer Class
 */
class DenseLayer {
public:
    DenseLayer();
    /**
     * DenseLayer constructor, allocates random weights/biases and gradient matrices/vectors
     *
     * @param numNeurons number of neurons in the layer
     * @param numInputs number of inputs to the layer
     * @param lr learning rate hyperparameter of layer
     * @param m momentum hyperparameter of layer
     * @param hidden if this layer is a hidden layer
     */
    DenseLayer(int numNeurons, int numInputs, double lr, double m, bool hidden);
    void loadLayer(matrix& weights, vector<double>& biases, int numNeurons, int numWeightsPerNeuron, double lr, double m, bool hidden);

    /**
     * Forward propagate data through the layer
     *
     * @param dataIn input vector
     * @return output vector (weighted sum through activation function)
     */
    vector<double> propagate(vector<double> dataIn);
    /**
     * Back propagation function to calculate weight/bias gradients
     *
     * @param nextDeltas deltas from next layer
     * @param nextWeights weights from next layer
     * @return neuron deltas
     */
    vector<double> backPropagate(const vector<double>& nextDeltas, const matrix& nextWeights={{{}}});
    /**
     * Same as backPropagate, but with mini-batches
     *
     * @param nextDeltas deltas from next layer
     * @param batchSize size of each batch
     * @param nextWeights weights from next layer
     * @return neuron deltas
     */
    vector<double> miniBatchBackPropagate(vector<double> nextDeltas, int batchSize, matrix nextWeights={{{}}});
    /**
     * Update each weight/bias using gradients
     */
    void update();
    /**
     * Returns true if layer is hidden, else false
     * @return bool indicating if hidden
     */

    void setLr(double lr);
    void setWeights(matrix& weights);

    void setBiases(vector<double> biases);
    bool getHidden();
    /**
     * Returns number of weights
     * @return number of weights
     */
    int getNumWeights() const;
    /**
     * Returns number of neurons
     * @return number of neurons
     */
    int getNumNeurons() const;
    /**
     * Returns matrix containing all weights of layer
     * @return layer weights matrix
     */
    matrix getWeights() const;
    /**
     * Returns vector containing all biases of layer
     * @return layer biases vector
     */
    vector<double> getBiases() const;
    /**
     * Resets all deltas for the layer to 0
     */
    void resetDeltas();

private:
    /**
     * Applies weighted sum to input vector
     * Results stored in preOutputs vector
     */
    void weightedSum();
    /**
     * Applies softmax activation function to preOutputs vector
     * Results stored in activatedOutputs vector
     */
    void softmax();
    /**
     * Calculates softmax derivative given next layer deltas
     * Results stored in gradients vector
     * @param nextDeltas next layer deltas
     */
    void softmaxDeriv(const vector<double>& nextDeltas);
    /**
     * Applies ReLu activation function to preOutputs vector
     * Results stored in activatedOutputs vector
     */
    void ReLu();
    /**
     * Calculates ReLu derivative given next layer weights and deltas
     * Results stored in gradients vector
     * @param nextDeltas next layer deltas
     * @param nextWeights next layer weights
     */
    void ReLuDeriv(const vector<double>& nextDeltas, const matrix& nextWeights);
    /**
     * Calculates derivative with respect to each weight and bias
     * Results stored in weightGradients and biasGradients vectors
     */
    void weightDeriv();
    /**
     * Same as weightDeriv but with mini-batches
     * @param batchSize size of each batch
     */
    void miniWeightDeriv(int batchSize);
    /**
     * Randomly generates a matrix of weights using Normalized Xavier Weight Initialization
     */
    void generateWeights();
    /**
     * Creates a random vector of size n given upper and lower bounds
     *
     * @param n size of vector
     * @param lower lower bound of random values
     * @param upper upper bound of random values
     * @return vector containing random values
     */
    vector<double> allocRandVector(int n, double lower, double upper);
    /**
     * Creates a random matrix of size r x c given upper and lower bounds
     *
     * @param r rows in matrix
     * @param c columns in matrix
     * @param lower lower bound of random values
     * @param upper upper bound of random values
     * @return matrix containing random values
     */
    matrix allocRandMatrix(int r, int c, double lower, double upper);

    //Private Variables
    matrix weights;
    matrix weightGradients;
    matrix prevWeightGradients;
    vector<double> biases;
    vector<double> biasGradients;
    vector<double> prevBiasGradients;
    vector<double> preOutputs;
    vector<double> activatedOutputs;
    vector<double> gradients;
    vector<double> inputs;
    int numNeurons;
    int numInputs;
    double lr;
    double m;
    bool hidden;
    // real number distribution in a range
    uniform_real_distribution<double> unif;
    // random engine
    default_random_engine rng;
};

/**
 * Convolution Layer Class
 */
class ConvLayer {
public:
    /**
     * Empty constructor
     */
    ConvLayer();
    /**
     * ConvLayer constructor, allocates random weight tensors and gradient tensors
     *
     * @param numKernels number of kernels in layer
     * @param kernelSize dimension of each kernel in layer
     * @param inputChannels number of input channels
     * @param inputDim dimension of input channels
     * @param lr learning rate hyperparameter of layer
     * @param m momentum hyperparameter of layer
     * @param padding if padding on layer inputs
     * @param poolType type of pooling
     */
    ConvLayer(int numKernels, int kernelDim, int inputChannels, int inputDim, double lr, double m, bool padding, int poolType);
    void loadLayer(vector<tensor>& kernels, int numKernels, int kernelDim, int inputChannels, int inputDim, double lr, double m, int padWidth, int poolType);
    /**
     * Forward propagates dataIn through layer
     *
     * @param dataIn input tensor
     * @return output tensor
     */
    tensor propagate(tensor dataIn);
    /**
     * Back propagation function to calculate weight gradients
     *
     * @param nextDeltas deltas from next layer
     * @return input tensor deltas
     */
    tensor backPropagate(const tensor& nextDeltas);
    /**
     * Same as backPropagate but with mini-batch
     *
     * @param nextDeltas deltas from next layer
     * @param batchSize size of batches
     * @return input tensor deltas
     */
    tensor miniBatchBackPropagate(const tensor& nextDeltas, int batchSize);
    /**
     * Update each weight with gradients
     */
    void update();
    /**
     * Returns vector of tensors containing all kernels of layer
     * @return layer kernels vector
     */
    void setLr(double lr);
    vector<tensor> getKernels();
    /**
     * Set value of all kernels of layer
     * @param kerns kernels to be set
     */
    void setKernels(vector<tensor> kerns);
    /**
     * Returns number of kernels in each channel
     * @return number of kernels in each channel
     */
    int getNumKernels() const;
    /**
     * Sets number of kernels in each channel
     * @param kCount number of kernels per channel
     */
    void setNumKernels(int kCount);
    /**
     * Returns pooling type
     * @return pooling type int
     */
    int getPoolType();
    /**
     * Returns input tensor dimensions
     * @return input tensor dimensions
     */
    int getInputDim() const;
    /**
     * Returns number of input channels
     * @return number of input channels
     */
    int getInputChannels() const;
    /**
     * Sets number of input channels
     * @param channelCount number of input channels
     */
    void setInputChannels(int channelCount);
    /**
     * Returns dimension of kernels in layer
     * @return dimension of kernels in layer
     */
    int getKernelDim() const;
    /**
     * Sets dimension of kernels in layer
     * @param kDim dimension of kernels in layer
     */
    void setKernelDim(int kDim);
    /**
     * Returns dimension of output kernels
     * @return dimension of output kernels
     */
    int getOutputDim() const;
    /**
     * Sets dimension of output kernels
     * @param oDim dimension of output kernels
     */
    void setOutputDim(int oDim);
    /**
     * Returns padding width applied to inputs of layer
     * @return padding width applied ot inputs of layer
     */
    int getPadWidth() const;
    /**
     * Sets padding width applied to inputs of layer
     * @param pWidth padding width applied to inputs of layer
     */
    void setPadWidth(int pWidth);
    /**
     * Resets all deltas in layer to 0
     */
    void resetDeltas();

private:
    void initLayer();
    /**
     * Updates kernel gradients
     */
    void updateKernels();
    /**
     * Same as updateKernels but with mini-batches
     * @param batchSize size of batches
     */
    void miniUpdateKernels(int batchSize);
    /**
     * Calculates derivatives of next layer deltas with respect to kernel weights
     */
    void kernelDeriv();
    /**
     * Calculates derivatives of next layer deltas with respect to layer inputs
     */
    void inputDeriv();
    /**
     * Applies ReLu activation function to kernelSumOutputs
     * Results stored in kernelActivatedOutputs
     */
    void ReLu();
    /**
     * Calculates derivatives with respect to each activated output
     * Results stored in activationDeltas
     */
    void ReLuDeriv();
    /**
     * Applies pooling to kernelActivatedOutputs
     * Results stored in outputs
     *
     * @param dataIn tensor to be pooled (kernelActivatedOutputs)
     */
    void pool(tensor dataIn);
    /**
     * Calculates pooling derivatives with respect to each pooled output
     *
     * @param deltas next layer deltas
     */
    void poolDeriv(const tensor& deltas);
    /**
     * Allocates a random r x c matrix
     *
     * @param r rows
     * @param c columns
     * @return r x c matrix
     */
    matrix allocRandomMatrix(int r, int c);
    /**
     * Allocates a random numKernels x dim x dim tensor
     *
     * @param numKernels number of kernels in tensor
     * @param dim dimension of each channel
     * @return random numKernels x dim x dim tensor
     */
    tensor allocRandomTensor(int numKernels, int dim) ;
    /**
     * Allocates a random vector of size(numChannels)
     *
     * @param numKernels number of kernels per channel
     * @param numChannels number of channels in vector
     * @param dim kernel dimensions
     * @return random vector of size(numKernels) containing numChannels x dim x dim tensors
     */
    vector<tensor> allocRandomTensVec(int numKernels, int numChannels, int dim);
    /**
     * Allocates a vector of size(numKernels) to contain pooling values
     *
     * @param numKernels number of kernels
     * @param dim dimension of kernels
     * @return vector of tensors to contain pooling values
     */
    vector<intTensor> allocPoolTensor(int numKernels, int dim);



    //Private Variables
    vector<tensor> kernels;
    tensor inputs;
    tensor inputDeltas;
    tensor poolDeltas;
    tensor activationDeltas;
    vector<tensor> kernelDeltas;
    vector<tensor> kernelGradients;
    vector<tensor> kernelPrevGradients;
    tensor kernelSumOutputs;
    tensor kernelActivatedOutputs;
    tensor outputs;
    // indices of max value from max pooling
    vector<intTensor> poolIndices;
    // real number distribution
    uniform_real_distribution<double> unif;
    // random engine
    default_random_engine rng;
    int kernelDim;
    int numKernels;
    int inputDim;
    int inputChannels;
    int kernelOutputDim;
    int outputDim;
    double lr;
    double m;
    bool padding;
    int padWidth = 0;
    // 0=None, 1=MaxPooling, 2=AvgPooling
    int poolType;
};

/******************************************************************************
 *  File: ConvNet.h
 *
 *  A header file for a Convolutional Neural Network Class
 *
 *  Last modified by: Alex Gilsoul
 *  Last modified on: Nov 2, 2022
 ******************************************************************************/

#pragma once

#include "../include/alg_stopwatch.h"
#include "../include/ConvLayer.h"
#include <fstream>
#include <thread>
#include <windows.h>

using std::make_shared;
using std::thread;
using std::setprecision;
using std::shared_ptr;
using std::ostream;
using std::ofstream;
using std::string;
using std::cout;
using std::endl;

/**
 * Convolutional Neural Network Class
 */
class ConvNet {
public:
    /**
     * Constructor for ConvNet class
     * just assigns private variables
     *
     * @param inputChannelCount number of input channels
     * @param inputDim dimension of input image
     * @param lr learning rate hyperparameter
     * @param m momentum hyperparameter
     */
    ConvNet(int inputChannelCount, int inputDim, double lr, double m=0.0);
    /**
     * Adds a convolution layer to the neural network
     *
     * @param numKernels number of kernels in the layer
     * @param kernelDim dimension of kernels in the layer
     * @param padding if padding will be applied
     * @param pooling if pooling will be applied
     * @param poolType type of pooling (average pooling is the default)
     */
    void addConvLayer(int numKernels, int kernelDim, bool padding=true, bool pooling=false, const string& poolType="avg");
    /**
     * Adds a dense layer to the neural network
     *
     * @param numNeurons number of neurons in the layer
     * @param hidden if layer is hidden (not hidden means output)
     */
    void addDenseLayer(int numNeurons, bool hidden);
    /**
     * Given an input tensor (ex: RGB image with one matrix for each R, G, and B) predicts the category of the image
     *
     * @param data input image
     * @return index of predicted category
     */
    int predict(tensor data);
    /**
     * Propagates an image through the network
     *
     * @param dataIn input image
     * @return vector of values from output layer of the network
     */
    vector<double> propagate(tensor dataIn);
    /**
     *
     * @param trainData
     * @param trainLabels
     * @param iterations
     * @param batchSize
     * @param verbose
     */
    void fitModel(vector<tensor> trainData, vector<int> trainLabels, int iterations, int batchSize=1, bool verbose=false);
    /**
     * One-hot encodes a vector of index labels assuming a range of [0, (output layer size) - 1]
     *
     * @param labels labels to be encoded
     * @return a vector containing vectors (matrix) of encoded labels
     */
    matrix oneHotEncode(vector<int> labels);
    /**
     * Evaluates the accuracy of the model
     *
     * @param testData testing images
     * @param testLabels testing labels
     * @return (correct predictions) / (total testing images)
     */
    double eval(vector<tensor> testData, vector<int> testLabels);
    /**
     * Prints information of the neural network to the console
     */
    void printNet();

    /*
     * file format:
     * learningRate,momentum,numConnections
     * numberConvolutionLayers,numberDenseLayers,inputDim,inputChannels
     * //for every conv layer:
     * numKernels,kernelDim,inputChannels,inputDim,padWidth,poolingType
     * *all kernels
     * //for every dense layer
     * numNeurons,numWeightsPerNeuron,Hidden?(T/F)
     * *all weights
     * *all biases
     */
    /**
     * Saves neural network weights and biases to text file to be loaded in later
     *
     * @param fileName name of file to save to
     * @return true if the file was successfully saved, else false
     */
    bool save(string fileName);


private:
    /**
     * Back propagation training algorithm
     *
     * @param trainData training images
     * @param trainLabels training labels (categories)
     * @param iterations number of iterations to train over
     */
    void backPropagate(vector<tensor> trainData, vector<int> trainLabels, int iterations);
    /**
     * Same as backPropagate, but uses mini-batches
     *
     * @param trainData training images
     * @param trainLabels training labels (categories)
     * @param iterations number of iterations to train over
     * @param batchSize size of batches
     */
    void miniBatchBackPropagate(vector<tensor> trainData, vector<int> trainLabels, int iterations, int batchSize);
    /**
     * Uses multithreading to print a bar indicating training progress
     */
    void progressBar();
    /**
     * Called by progressBar to actually print progress bar to console
     *
     * @param curVal current progress
     * @param goal progress goal
     * @param barWidth width of bar
     * @param watch Stopwatch object, used to time rotation of progress indicator
     */
    void printBar(int curVal, int goal, int barWidth, StopWatch watch);

    // Private Variables
    // Vector containing pointers to all convolution layers
    vector<shared_ptr<ConvLayer>> convLayers;
    // Vector containing pointers to all dense layers
    vector<shared_ptr<DenseLayer>> denseLayers;
    // Number of input channels
    int inputChannelCount;
    // Dimension of inputs
    int inputDim;
    // Current progress for progress bar
    shared_ptr<double> curProgress = make_shared<double>(0.0);
    // Progress goal for progress bar
    shared_ptr<double> progressGoal = make_shared<double>(0.0);
    // Progress bar width
    int barSize = 70;
    // Training status
    shared_ptr<bool> doneTraining = make_shared<bool>(false);
    // Loading indicator strings
    string loading[4] = {" | ", " / ", " - ", " \\ "};
    // Learning rate
    double lr;
    // Momentum
    double m;
    // Number of neuron connections
    int numConnections = 0;
};

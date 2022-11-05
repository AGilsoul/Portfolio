//
// Created by agils on 11/2/2022.
//

#include "../include/ConvLayer.h"

// General methods
void multTensors(tensor& t1, tensor& t2) {
    for (int channelIndex = 0; channelIndex < t1.size(); channelIndex++) {
        for (int row = 0; row < t1[channelIndex].size(); row++) {
            for (int col = 0; col < t1[channelIndex][row].size(); col++) {
                t1[channelIndex][row][col] *= t2[channelIndex][row][col];
            }
        }
    }
}

vector<double> flattenTensor(tensor dataIn) {
    vector<double> res(dataIn.size() * dataIn[0].size() * dataIn[0][0].size());

    for (int channelIndex = 0; channelIndex < dataIn.size(); channelIndex++) {
        for (int row = 0; row < dataIn[channelIndex].size(); row++) {
            for (int col = 0; col < dataIn[channelIndex][row].size(); col++) {
                int index = row * dataIn.size() * dataIn[0][0].size() + (col * dataIn.size()) + channelIndex;
                res[index] = dataIn[channelIndex][row][col];
            }
        }
    }
    return res;
}

tensor reshapeVector(vector<double> vec, int channelCount, int r, int c) {
    if (channelCount * r * c != vec.size()) {
        throw exception("Invalid sizes");
    }

    tensor res = allocTensor(channelCount, r, c);
    int indexCount = 0;

    for (int channelIndex = 0; channelIndex < channelCount; channelIndex++) {
        for (int row = 0; row < r; row++) {
            for (int col = 0; col < c; col++) {
                res[channelIndex][row][col] = vec[indexCount];
                indexCount++;
            }
        }
    }
    return res;
}

tensor addPadding(tensor dataIn, int padWidth) {
    // dimension of padded output
    int newDimension = dataIn[0].size() + 2 * padWidth;
    // number of channels
    int numChannels = dataIn.size();
    // padded result tensor
    tensor paddedRes(numChannels);
    // zero vector for padding
    vector<double> zeroVector(newDimension, 0.0);
    // for every channel
    for (int channelIndex = 0; channelIndex < dataIn.size(); channelIndex++) {
        // new channel to be added to padded result tensor
        matrix newChannel(newDimension);
        // add top padding
        for (int padIndex = 0; padIndex < padWidth; padIndex++) {
            newChannel[padIndex] = zeroVector;
        }
        // add side padding
        for (int row = 0; row < dataIn[channelIndex].size(); row++) {
            // current row to be added to current channel
            vector<double> currentRow(newDimension, 0.0);
            for (int col = 0; col < dataIn[channelIndex][row].size(); col++) {
                // update current row column values
                currentRow[col + padWidth] = dataIn[channelIndex][row][col];
            }
            // update new channel row
            newChannel[row + padWidth] = currentRow;
        }
        // add bottom padding
        for (int padIndex = newDimension - padWidth; padIndex < newDimension; padIndex++) {
            newChannel[padIndex] = zeroVector;
        }
        // update padded result channel
        paddedRes[channelIndex] = newChannel;
    }
    // return padded result
    return paddedRes;
}

tensor delPadding(tensor dataIn, int padWidth) {
    // dimension of output with padding removed
    int newDimension = dataIn[0].size() - 2 * padWidth;
    // number of channels
    int numChannels = dataIn.size();
    // result tensor with padding removed
    tensor unpaddedRes(numChannels);
    // for every channel
    for (int channelIndex = 0; channelIndex < dataIn.size(); channelIndex++) {
        // current channel
        matrix newChannel(newDimension);
        // add side padding
        for (int row = padWidth; row < dataIn[channelIndex].size() - padWidth; row++) {
            // current row
            vector<double> currentRow(newDimension);
            // update columns in current row
            for (int col = padWidth; col < dataIn[channelIndex][row].size() - padWidth; col++) {
                currentRow[col - padWidth] = dataIn[channelIndex][row][col];
            }
            // update current channel row
            newChannel[row - padWidth] = (currentRow);
        }
        // update result tensor with channel
        unpaddedRes[channelIndex] = newChannel;
    }
    // return result tensor
    return unpaddedRes;
}

matrix allocMatrix(int r, int c, double val) {
    matrix res(r);
    for (int row = 0; row < r; row++) {
        res[row] = vector<double>(c, val);
    }
    return res;
}

tensor allocTensor(int channelCount, int dim, double value) {
    tensor resTensor(channelCount);
    for (auto & channelIndex : resTensor) {
        channelIndex = allocMatrix(dim, dim, value);
    }
    return resTensor;
}

vector<tensor> allocTensVec(int numTensors, int channelCount, int dim, double value) {
    vector<tensor> tensors(numTensors);
    for (auto & tensIndex: tensors) {
        tensIndex = allocTensor(channelCount, dim, value);
    }
    return tensors;
}

void normalize(vector<tensor>& dataIn, vector<double> minMaxRanges) {
    // for data point channel
    for (int d = 0; d < dataIn.size(); d++) {
        for (int channelIndex = 0; channelIndex < dataIn[d].size(); channelIndex++) {
            // for every data trait
            double min = minMaxRanges[0];
            double max = minMaxRanges[1];
            for (int row = 0; row < dataIn[d][channelIndex].size(); row++) {
                // gets the min and max values for the data range
                // for every data point x in the dataset
                for (int col = 0; col < dataIn[d][channelIndex][row].size(); col++) {
                    dataIn[d][channelIndex][row][col] = (dataIn[d][channelIndex][row][col] - min) / (max - min);
                }
            }
        }
    }
}


// Hidden Layer Methods
DenseLayer::DenseLayer(int numNeurons, int numInputs, double lr, double m, bool hidden): numNeurons(numNeurons), numInputs(numInputs), lr(lr), m(m), hidden(hidden) {
    generateWeights();
    this->prevWeightGradients = allocMatrix(numNeurons, numInputs);
    this->prevBiasGradients = vector<double>(numNeurons);
    this->prevBiasGradients = vector<double>(numNeurons);
    this->activatedOutputs = vector<double>(numNeurons);
    resetDeltas();
}

vector<double> DenseLayer::propagate(vector<double> dataIn) {
    this->inputs = dataIn;
    weightedSum();
    if (hidden) {
        ReLu();
    }
    else {
        softmax();
    }
    return this->activatedOutputs;
}

vector<double> DenseLayer::backPropagate(const vector<double>& nextDeltas, const matrix& nextWeights) {
    // calculate input deltas first and then calculate weight deltas
    if (hidden) {
        ReLuDeriv(nextDeltas, nextWeights);
    }
    else {
        softmaxDeriv(nextDeltas);
    }
    weightDeriv();
    return this->gradients;
}

vector<double> DenseLayer::miniBatchBackPropagate(vector<double> nextDeltas, int batchSize, matrix nextWeights) {
    // calculate input deltas first and then calculate weight deltas
    if (hidden) {
        ReLuDeriv(std::move(nextDeltas), nextWeights);
    }
    else {
        softmaxDeriv(std::move(nextDeltas));
    }
    miniWeightDeriv(batchSize);
    return this->gradients;
}

void DenseLayer::update() {
    for (int n = 0; n < numNeurons; n++) {
        for (int w = 0; w < numInputs; w++) {
            this->weights[n][w] -= this->weightGradients[n][w];
            this->prevWeightGradients[n][w] = this->weightGradients[n][w];
        }
        this->biases[n] -= this->biasGradients[n];
        this->prevBiasGradients[n] = this->biasGradients[n];
    }
}

bool DenseLayer::getHidden() {
    return hidden;
}

int DenseLayer::getNumWeights() const {
    return numInputs;
}

int DenseLayer::getNumNeurons() const {
    return numNeurons;
}

matrix DenseLayer::getWeights() const {
    return weights;
}

vector<double> DenseLayer::getBiases() const {
    return biases;
}

void DenseLayer::resetDeltas() {
    this->weightGradients = allocMatrix(numNeurons, numInputs);
    this->biasGradients = vector<double>(numNeurons);
    this->gradients = vector<double>(numNeurons);
}

void DenseLayer::weightedSum() {
    this->preOutputs = vector<double>(numNeurons);
    for (int n = 0; n < numNeurons; n++) {
        for (int w = 0; w < numInputs; w++) {
            this->preOutputs[n] += weights[n][w] * inputs[w];
        }
        this->preOutputs[n] += biases[n];
    }
}

void DenseLayer::softmax() {
    double denom = 0;
    for (int n = 0; n < numNeurons; n++) {
        denom += exp(this->preOutputs[n]);
    }
    for (int n = 0; n < numNeurons; n++) {
        this->activatedOutputs[n] = exp(this->preOutputs[n]) / denom;
    }
}

void DenseLayer::softmaxDeriv(const vector<double>& nextDeltas) {
    for (int n = 0; n < numNeurons; n++) {
        this->gradients[n] = this->activatedOutputs[n] - nextDeltas[n];
    }
}

void DenseLayer::ReLu() {
    for (int n = 0; n < numNeurons; n++) {
        if (preOutputs[n] > 0) {
            this->activatedOutputs[n] = preOutputs[n];
        }
        else {
            this->activatedOutputs[n] = 0.0;
        }
    }
}

void DenseLayer::ReLuDeriv(const vector<double>& nextDeltas, const matrix& nextWeights) {
    for (int n = 0; n < numNeurons; n++) {
        if (preOutputs[n] > 0) {
            double total = 0;
            for (int nextN = 0; nextN < nextWeights.size(); nextN++) {
                total += nextWeights[nextN][n] * nextDeltas[nextN];
            }
            this->gradients[n] = total;
        }
        else {
            this->gradients[n] = 0;
        }
    }
}

void DenseLayer::weightDeriv() {
    for (int n = 0; n < numNeurons; n++) {
        for (int w = 0; w < numInputs; w++) {
            this->weightGradients[n][w] = this->inputs[w] * this->gradients[n] * this->lr + this->prevWeightGradients[n][w] * this->m;
        }
        this->biasGradients[n] = this->gradients[n] * this->lr + this->prevBiasGradients[n] * this->m;
    }
}

void DenseLayer::miniWeightDeriv(int batchSize) {
    for (int n = 0; n < numNeurons; n++) {
        for (int w = 0; w < numInputs; w++) {
            this->weightGradients[n][w] += (this->inputs[w] * this->gradients[n] * this->lr + this->prevWeightGradients[n][w] * this->m) / batchSize;
        }
        this->biasGradients[n] += (this->gradients[n] * this->lr + this->prevBiasGradients[n] * this->m) / batchSize;
    }
}

void DenseLayer::generateWeights() {
    double weightLimit = (sqrt(6.0) / sqrt(numInputs + numNeurons));
    weights = allocRandMatrix(numNeurons, numInputs, -weightLimit, weightLimit);
    biases = allocRandVector(numNeurons, -weightLimit, weightLimit);
}

vector<double> DenseLayer::allocRandVector(int n, double lower, double upper) {
    unif = uniform_real_distribution<double>(lower, upper);
    rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
    vector<double> res(n);
    for (int i = 0; i < n; i++) {
        res[i] = unif(rng);
    }
    return res;
}

matrix DenseLayer::allocRandMatrix(int r, int c, double lower, double upper) {
    unif = uniform_real_distribution<double>(lower, upper);
    rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
    matrix res(r);
    for (int row = 0; row < r; row++) {
        vector<double> vec(c);
        for (int col = 0; col < c; col++) {
            vec[col] = unif(rng);
        }
        res[row] = vec;
    }
    return res;
}

// Convolution Layer Methods
ConvLayer::ConvLayer() {}

ConvLayer::ConvLayer(int numKernels, int kernelSize, int inputChannels, int inputDim, double lr, double m, bool padding, int poolType): numKernels(numKernels), kernelSize(kernelSize), inputChannels(inputChannels), inputDim(inputDim), lr(lr), m(m), padding(padding), poolType(poolType) {
    // set up rng
    this->unif = uniform_real_distribution<double>(-1, 1);
    this->rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
    // set pad width for kernel input padding
    if (padding) {
        this->padWidth = ceil(double(kernelSize) / 2);
    }
    else {
        this->padWidth = 0;
    }
    // calc output dimensions of kernel matrices
    this->kernelOutputDim = (inputDim + 2 * padWidth - kernelSize) + 1;
    // re-calc output dimensions if pooling enabled
    if (poolType != 0) {
        // if output dimension divisible by 2
        if (this->kernelOutputDim % 2 != 0) {
            this->outputDim = int((this->kernelOutputDim + 1) / 2);
        }
            // else
        else {
            this->outputDim = int(this->kernelOutputDim / 2);
        }
        // allocate space for pool indexing tensor
        poolIndices = allocPoolTensor(numKernels, this->outputDim);
        this->outputs = allocTensor(numKernels, this->outputDim);
    }
    else {
        this->outputDim = kernelOutputDim;
    }
    this->kernelPrevGradients = allocTensVec(numKernels, inputChannels, this->kernelSize);
    // allocate activation deltas
    this->activationDeltas = allocTensor(numKernels, this->kernelOutputDim);
    // allocate space for output kernel output tensors
    this->kernelSumOutputs = allocTensor(numKernels, this->kernelOutputDim);
    this->kernelActivatedOutputs = allocTensor(numKernels, this->kernelOutputDim);
    // generate kernel values
    this->kernels = allocRandomTensVec(numKernels, inputChannels, kernelSize);
    resetDeltas();
    this->unif = uniform_real_distribution<double>(0, 1);
}

tensor ConvLayer::propagate(tensor dataIn) {
    this->kernelSumOutputs = allocTensor(numKernels, this->kernelOutputDim);
    // adjust input with padding if enabled
    if (this->padding) {
        dataIn = addPadding(dataIn, padWidth);
    }
    this->inputs = dataIn;
    this->inputDim = dataIn[0].size();
    // for every kernel
    for (int kernelIndex = 0; kernelIndex < this->numKernels; kernelIndex++) {
        // for each channel in the kernel
        for (int channelIndex = 0; channelIndex < this->inputChannels; channelIndex++) {
            // for every row in the output
            for (int row = 0; row < this->kernelOutputDim; row++) {
                // for every col in the output
                for (int col = 0; col < this->kernelOutputDim; col++) {
                    // get kernel sums
                    for (int kRow = 0; kRow < this->kernelSize; kRow++) {
                        for (int kCol = 0; kCol < this->kernelSize; kCol++) {
                            this->kernelSumOutputs[kernelIndex][row][col] += dataIn[channelIndex][row + kRow][col + kCol] * this->kernels[kernelIndex][channelIndex][kRow][kCol];
                        }
                    }
                }
            }
        }
    }

    this->ReLu();
    if (poolType == 0) {
        this->outputs = kernelActivatedOutputs;
    }
    else {
        this->pool(kernelActivatedOutputs);
    }
    return this->outputs;
}

tensor ConvLayer::backPropagate(const tensor& nextDeltas) {
    // if pooling layer present
    if (poolType != 0) {
        poolDeriv(nextDeltas);
    }
        // if no pooling layers
    else {
        this->poolDeltas = nextDeltas;
    }

    // calculate activation function delta
    ReLuDeriv();
    // chain rule with previous delta
    multTensors(this->activationDeltas, this->poolDeltas);
    // calculate input derivatives
    inputDeriv();
    // calculate kernel derivatives and apply gradients
    kernelDeriv();
    updateKernels();
    return this->inputDeltas;
}

tensor ConvLayer::miniBatchBackPropagate(const tensor& nextDeltas, int batchSize) {
    // if pooling layer present
    if (poolType != 0) {
        poolDeriv(nextDeltas);
    }
        // if no pooling layers
    else {
        this->poolDeltas = nextDeltas;
    }

    // calculate activation function delta
    ReLuDeriv();
    // chain rule with previous delta
    multTensors(this->activationDeltas, this->poolDeltas);
    // calculate input derivatives
    inputDeriv();
    // calculate kernel derivatives and apply gradients
    kernelDeriv();
    miniUpdateKernels(batchSize);
    return this->inputDeltas;
}

void ConvLayer::update() {
    for (int kernelIndex = 0; kernelIndex < numKernels; kernelIndex++) {
        for (int channelIndex = 0; channelIndex < inputChannels; channelIndex++) {
            for (int row = 0; row < kernelSize; row++) {
                for (int col = 0; col < kernelSize; col++) {
                    this->kernels[kernelIndex][channelIndex][row][col] -= this->kernelGradients[kernelIndex][channelIndex][row][col];
                    this->kernelPrevGradients[kernelIndex][channelIndex][row][col] = this->kernelGradients[kernelIndex][channelIndex][row][col];
                }
            }
        }
    }
}

vector<tensor> ConvLayer::getKernels() {
    return kernels;
}

void ConvLayer::setKernels(vector<tensor> kerns) {
    kernels = kerns;
}

int ConvLayer::getNumKernels() const {
    return numKernels;
}

void ConvLayer::setNumKernels(int kCount) {
    numKernels = kCount;
}

int ConvLayer::getPoolType() {
    return poolType;
}

int ConvLayer::getInputDim() const {
    return inputDim;
}

int ConvLayer::getInputChannels() const {
    return inputChannels;
}

void ConvLayer::setInputChannels(int channelCount) {
    inputChannels = channelCount;
}

int ConvLayer::getKernelDim() const {
    return kernelSize;
}

void ConvLayer::setKernelDim(int kDim) {
    kernelSize = kDim;
}

int ConvLayer::getOutputDim() const {
    return outputDim;
}

void ConvLayer::setOutputDim(int oDim) {
    outputDim = oDim;
}

int ConvLayer::getPadWidth() const {
    return padWidth;
}

void ConvLayer::setPadWidth(int pWidth) {
    padWidth = pWidth;
}

void ConvLayer::resetDeltas() {
    kernelGradients = allocTensVec(numKernels, inputChannels, this->kernelSize);
}

void ConvLayer::updateKernels() {
    for (int kernelIndex = 0; kernelIndex < numKernels; kernelIndex++) {
        for (int channelIndex = 0; channelIndex < inputChannels; channelIndex++) {
            for (int row = 0; row < kernelSize; row++) {
                for (int col = 0; col < kernelSize; col++) {
                    this->kernelGradients[kernelIndex][channelIndex][row][col] = this->kernelDeltas[kernelIndex][channelIndex][row][col] * lr + this->kernelPrevGradients[kernelIndex][channelIndex][row][col] * m;
                }
            }
        }
    }
}

void ConvLayer::miniUpdateKernels(int batchSize) {
    for (int kernelIndex = 0; kernelIndex < numKernels; kernelIndex++) {
        for (int channelIndex = 0; channelIndex < inputChannels; channelIndex++) {
            for (int row = 0; row < kernelSize; row++) {
                for (int col = 0; col < kernelSize; col++) {
                    this->kernelGradients[kernelIndex][channelIndex][row][col] += this->kernelDeltas[kernelIndex][channelIndex][row][col] / batchSize;
                }
            }
        }
    }
}

void ConvLayer::kernelDeriv() {
    this->kernelDeltas = allocTensVec(numKernels, inputChannels, this->kernelSize);
    for (int kernelIndex = 0; kernelIndex < numKernels; kernelIndex++) {
        for (int channelIndex = 0; channelIndex < inputChannels; channelIndex++) {
            for (int row = 0; row < kernelOutputDim; row++) {
                for (int col = 0; col < kernelOutputDim; col++) {
                    // if deltas are non-zero
                    if (this->activationDeltas[kernelIndex][row][col] != 0) {
                        // add all values in the kernel range
                        for (int kRow = 0; kRow < kernelSize; kRow++) {
                            for (int kCol = 0; kCol < kernelSize; kCol++) {
                                this->kernelDeltas[kernelIndex][channelIndex][kRow][kCol] += this->activationDeltas[kernelIndex][row][col] * this->inputs[channelIndex][row + kRow][col + kCol];
                            }
                        }
                    }
                }
            }
        }
    }
}

void ConvLayer::inputDeriv() {
    this->inputDeltas = allocTensor(inputChannels, inputDim);
    for (int kernelIndex = 0; kernelIndex < numKernels; kernelIndex++) {
        for (int row = 0; row < kernelOutputDim; row++) {
            for (int col = 0; col < kernelOutputDim; col++) {
                if (this->activationDeltas[kernelIndex][row][col] != 0) {
                    for (int channelIndex = 0; channelIndex < inputChannels; channelIndex++) {
                        for (int kRow = 0; kRow < kernelSize; kRow++) {
                            for (int kCol = 0; kCol < kernelSize; kCol++) {
                                this->inputDeltas[channelIndex][row + kRow][col + kCol] += this->kernels[kernelIndex][channelIndex][kRow][kCol] * this->activationDeltas[kernelIndex][row][col];
                            }
                        }
                    }

                }
            }
        }
    }
}

void ConvLayer::ReLu() {
    for (int kernelIndex = 0; kernelIndex < this->numKernels; kernelIndex++) {
        for (int row = 0; row < this->kernelOutputDim; row++) {
            for (int col = 0; col < this->kernelOutputDim; col++) {
                this->kernelActivatedOutputs[kernelIndex][row][col] = max({0.0, this->kernelSumOutputs[kernelIndex][row][col]});
            }
        }
    }
}

void ConvLayer::ReLuDeriv() {
    for (int kernelIndex = 0; kernelIndex < this->numKernels; kernelIndex++) {
        for (int row = 0; row < this->kernelOutputDim; row++) {
            for (int col = 0; col < this->kernelOutputDim; col++) {
                if (this->kernelSumOutputs[kernelIndex][row][col] > 0) {
                    this->activationDeltas[kernelIndex][row][col] = 1;
                }
                else {
                    this->activationDeltas[kernelIndex][row][col] = 0;
                }
            }
        }
    }
}

void ConvLayer::pool(tensor dataIn) {
    // adjust dimensions of input data for pooling
    int d = this->kernelOutputDim;
    if (d % 2 != 0) {
        vector<double> zeroVec(d + 1, 0.0);
        for (int kernelIndex = 0; kernelIndex < numKernels; kernelIndex++) {
            for (int row = 0; row < d; row++) {
                dataIn[kernelIndex][row].push_back(0.0);
            }
            dataIn[kernelIndex].push_back(zeroVec);
        }
    }

    // if pooling type is max pooling
    if (this->poolType == 1) {
        // for every channel in pool output
        for (int kernelIndex = 0; kernelIndex < this->numKernels; kernelIndex++) {
            // for every row in pool output
            for (int row = 0; row < this->outputDim; row++) {
                // for every col in pool output
                for (int col = 0; col < this->outputDim; col++) {
                    bool varAssigned = false;
                    vector<int> maxIndices = {0, 0};
                    double maxVal = 0;
                    for (int fieldRow = 0; fieldRow < 2; fieldRow++) {
                        for (int fieldCol = 0; fieldCol < 2; fieldCol++) {
                            int rowIndex = row * 2 + fieldRow;
                            int colIndex = col * 2 + fieldCol;
                            double val = dataIn[kernelIndex][rowIndex][colIndex];
                            if (val > maxVal || !varAssigned) {
                                maxVal = val;
                                maxIndices = {rowIndex, colIndex};
                                varAssigned = true;
                            }
                        }
                    }
                    this->poolIndices[kernelIndex][row][col] = maxIndices;
                    this->outputs[kernelIndex][row][col] = maxVal;
                }
            }
        }
    }
    else if (this->poolType == 2) {
        // for every channel in pool output
        for (int kernelIndex = 0; kernelIndex < this->numKernels; kernelIndex++) {
            // for every row in pool output
            for (int row = 0; row < this->outputDim; row++) {
                // for every col in pool output
                for (int col = 0; col < this->outputDim; col++) {
                    double total = 0;
                    for (int fieldRow = 0; fieldRow < 2; fieldRow++) {
                        for (int fieldCol = 0; fieldCol < 2; fieldCol++) {
                            double val = dataIn[kernelIndex][row * 2 + fieldRow][col * 2 + fieldCol];
                            total += val;
                        }
                    }
                    this->outputs[kernelIndex][row][col] = total / 4.0;
                }
            }
        }
    }
}

void ConvLayer::poolDeriv(const tensor& deltas) {
    // if pool type is max-pooling
    if (poolType == 1) {
        this->poolDeltas = allocTensor(numKernels, this->kernelOutputDim);
        for (int kernelIndex = 0; kernelIndex < this->numKernels; kernelIndex++) {
            for (int row = 0; row < this->outputDim; row++) {
                for (int col = 0; col < this->outputDim; col++) {
                    vector<int> curPoolIndex = this->poolIndices[kernelIndex][row][col];
                    this->poolDeltas[kernelIndex][curPoolIndex[0]][curPoolIndex[1]] = deltas[kernelIndex][row][col];
                }
            }
        }
    }
        // if pool type is average pooling
    else if (poolType == 2) {
        this->poolDeltas = allocTensor(numKernels, this->kernelOutputDim, 0.25);
        for (int kernelIndex = 0; kernelIndex < this->numKernels; kernelIndex++) {
            for (int row = 0; row < this->kernelOutputDim; row++) {
                for (int col = 0; col < this->kernelOutputDim; col++) {
                    this->poolDeltas[kernelIndex][row][col] *= deltas[kernelIndex][int(row / 2)][int(col / 2)];
                }
            }
        }
    }
}

tensor ConvLayer::allocRandomTensor(int numKernels, int dim) {
    tensor resTensor(numKernels);
    for (auto & kernelIndex : resTensor) {
        matrix curChannel(dim);
        for (auto & row : curChannel) {
            vector<double> curRow(dim);
            for (double & col : curRow) {
                col = unif(rng);
            }
            row = curRow;
        }
        kernelIndex = curChannel;
    }
    return resTensor;
}

vector<tensor> ConvLayer::allocRandomTensVec(int numChannels, int numKernels, int dim) {
    vector<tensor> resTensorVec(numChannels);
    for (auto & tensIndex : resTensorVec) {
        tensor curTensor(numKernels);
        for (auto & matIndex : curTensor) {
            matrix curMat(dim);
            for (auto & row: curMat) {
                vector<double> curRow(dim);
                for (double & col : curRow) {
                    col = unif(rng);
                }
                row = curRow;
            }
            matIndex = curMat;
        }
        tensIndex = curTensor;
    }
    return resTensorVec;
}

vector<vector<vector<vector<int>>>> ConvLayer::allocPoolTensor(int numKernels, int dim) {
    vector<vector<vector<vector<int>>>> resTensor(numKernels);
    for (auto & channelIndex : resTensor) {
        vector<vector<vector<int>>> curChannel(dim);
        for (auto & row : curChannel) {
            vector<vector<int>> curRow(dim);
            for (auto & col : curRow) {
                vector<int> curCol(2);
                col = curCol;
            }
            row = curRow;
        }
        channelIndex = curChannel;
    }
    return resTensor;
}

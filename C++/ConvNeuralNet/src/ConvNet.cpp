//
// Created by agils on 11/2/2022.
//

#include "../include/ConvNet.h"


ConvNet::ConvNet(int inputChannelCount, int inputDim, double lr, double m): inputChannelCount(inputChannelCount), inputDim(inputDim), lr(lr), m(m) {}

void ConvNet::addConvLayer(int numKernels, int kernelDim, bool padding, bool pooling, const string& poolType) {
    int poolInt = 0;
    if (pooling) {
        if (poolType == "max") {
            poolInt = 1;
        }
        else if (poolType == "avg") {
            poolInt = 2;
        }
    }
    if (convLayers.empty()) {
        numConnections += numKernels * pow(kernelDim, 2) * inputChannelCount;
        convLayers.push_back(make_shared<ConvLayer>(numKernels, kernelDim, inputChannelCount, inputDim, lr, m, padding, poolInt));
    }
    else {
        numConnections += numKernels * pow(kernelDim, 2) * convLayers[convLayers.size() - 1]->getNumKernels();
        convLayers.push_back(make_shared<ConvLayer>(numKernels, kernelDim, convLayers[convLayers.size() - 1]->getNumKernels(), convLayers[convLayers.size() - 1]->getOutputDim(), lr, m, padding, poolInt));
    }

}

void ConvNet::addDenseLayer(int numNeurons, bool hidden) {
    if (convLayers.empty()) {
        throw exception("Add kernels before dense layers");
    }

    if (denseLayers.empty()) {
        int numInputs = convLayers[convLayers.size() - 1]->getNumKernels() * pow(convLayers[convLayers.size() - 1]->getOutputDim(), 2);
        denseLayers.push_back(make_shared<DenseLayer>(numNeurons, numInputs, lr, m, hidden));
        numConnections += numNeurons * numInputs;
    }
    else {
        int numInputs = denseLayers[denseLayers.size() - 1]->getNumNeurons();
        denseLayers.push_back(make_shared<DenseLayer>(numNeurons, numInputs, lr, m, hidden));
        numConnections += numNeurons * numInputs;
    }
}

int ConvNet::predict(tensor data) {
    auto res = propagate(data);
    int maxIndex = -1;
    double maxVal = 0;
    for (int i = 0; i < res.size(); i++) {
        if (res[i] > maxVal) {
            maxVal = res[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

vector<double> ConvNet::propagate(tensor dataIn) {
    for (auto & convLayer : convLayers) {
        dataIn = convLayer->propagate(dataIn);
    }
    vector<double> flatData = flattenTensor(dataIn);
    for (auto & denseLayer : denseLayers) {
        flatData = denseLayer->propagate(flatData);
    }
    return flatData;
}

void ConvNet::fitModel(vector<tensor> trainData, vector<int> trainLabels, int iterations, int batchSize, bool verbose) {
    int valSize = floor(trainData.size() / iterations);
    for (int i = 0; i < iterations; i++) {
        // create validation sets for testing purposes
        vector<tensor> curTrainData;
        vector<int> curTrainLabels;
        vector<tensor> valData;
        vector<int> valLabels;
        for (int d = 0; d < trainData.size(); d++) {
            if (d >= valSize * i && d < valSize * i + valSize) {
                valData.push_back(trainData[d]);
                valLabels.push_back(trainLabels[d]);
            }
            else {
                curTrainData.push_back(trainData[d]);
                curTrainLabels.push_back(trainLabels[d]);
            }
        }

        // set progress bar params
        *curProgress = 0;
        *progressGoal = 1;
        *doneTraining = false;
        if (verbose) { cout << "Iteration " << i + 1 << " Training Progress:" << endl; };
        thread thread_obj;
        // stochastic gradient descent thread
        if (batchSize == 1) {
            cout << "SGD" << endl;
            thread_obj = thread(&ConvNet::backPropagate, *this, curTrainData, curTrainLabels, 1);
        }
            // mini-batch gradient descent thread
        else {
            cout << "Mini-Batch" << endl;
            thread_obj = thread(&ConvNet::miniBatchBackPropagate, *this, curTrainData, curTrainLabels, 1, batchSize);
        }
        // progress bar thread
        if (verbose) {
            thread thread_progress(&ConvNet::progressBar, *this);
            thread_progress.join();
        }
        // join threads
        thread_obj.join();
        // validation accuracy
        cout << "Evaluating..." << endl;
        auto res = eval(valData, valLabels) * 100;
        cout << "Validation Accuracy: " << setprecision(4) << res << "%" << endl;

    }
}

matrix ConvNet::oneHotEncode(vector<int> labels) {
    int numLabels = denseLayers[denseLayers.size()-1]->getNumNeurons();
    matrix res(labels.size());
    for (int lIndex = 0; lIndex < res.size(); lIndex++) {
        vector<double> newVec(numLabels);
        newVec[labels[lIndex]] = 1;
        res[lIndex] = newVec;
    }
    return res;
}

double ConvNet::eval(vector<tensor> testData, vector<int> testLabels) {
    double numCorrect = 0;
    for (int d = 0; d < testData.size(); d++) {
        auto res = predict(testData[d]);
        if (res == testLabels[d]) {
            numCorrect++;
        }
    }
    return numCorrect / testData.size();
}

void ConvNet::printNet() {
    cout << "Convolutional Neural Network" << endl;
    cout << "____________________________________________________________________" << endl;
    cout << "Convolution Layers: " << endl;
    for (int i = 0; i < convLayers.size(); i++) {
        printf("Convolution Layer %d - Kernels: %d\tKernel Dimensions: %d\n", i + 1, convLayers[i]->getNumKernels(), convLayers[i]->getKernelDim());
    }
    cout << "____________________________________________________________________" << endl;
    cout << "Dense Layers: " << endl;
    for (int i = 0; i < denseLayers.size(); i++) {
        printf("Dense Layer %d - Neurons: %d\n", i + 1, denseLayers[i]->getNumNeurons());
    }
    printf("Num Connections: %d\n", numConnections);
}

bool ConvNet::save(string fileName) {
    try {
        ofstream saveFile(fileName);
        if (saveFile) {
            saveFile << lr << "," << m << "," << numConnections << endl;
            saveFile << convLayers.size() << "," << denseLayers.size() << "," << inputDim << "," << inputChannelCount << endl;
            // for every convolution layer
            for (int cLayer = 0; cLayer < convLayers.size(); cLayer++) {
                // current conv layer
                shared_ptr<ConvLayer> curLayer = convLayers[cLayer];
                saveFile << curLayer->getNumKernels() << "," << curLayer->getKernelDim() << "," << curLayer->getInputChannels() << "," << curLayer->getInputDim() << "," << curLayer->getPadWidth() << "," << curLayer->getPoolType() << endl;
                // current layer kernels
                vector<tensor> curKernels = curLayer->getKernels();
                // for each kernel tensor in the layer
                for (int kCount = 0; kCount < curLayer->getNumKernels(); kCount++) {
                    // for each channel matrix in the kernel
                    for (int cCount = 0; cCount < curLayer->getInputChannels(); cCount++) {
                        for (int row = 0; row < curLayer->getKernelDim(); row++) {
                            saveFile << curKernels[kCount][cCount][row][0];
                            for (int col = 1; col < curLayer->getKernelDim(); col++) {
                                saveFile << "," << curKernels[kCount][cCount][row][col];
                            }
                            saveFile << endl;
                        }
                    }
                }
            }

            // for every dense layer
            for (int dLayer = 0; dLayer < denseLayers.size(); dLayer++) {
                // current layer
                shared_ptr<DenseLayer> curLayer = denseLayers[dLayer];
                saveFile << curLayer->getNumNeurons() << "," << curLayer->getNumWeights() << "," << int(curLayer->getHidden()) << endl;
                // current layer weights and biases
                matrix curWeights = curLayer->getWeights();
                vector<double> curBiases = curLayer->getBiases();
                // for each neuron in the current layer
                for (int nCount = 0; nCount < curLayer->getNumNeurons(); nCount++) {
                    // for each weight in the current neuron
                    saveFile << curWeights[nCount][0];
                    for (int wCount = 1; wCount < curLayer->getNumWeights(); wCount++) {
                        saveFile << "," << curWeights[nCount][wCount];
                    }
                    saveFile << endl;
                }
                // for each neuron bias in the current layer
                saveFile << curBiases[0];
                for (int bCount = 1; bCount < curLayer->getNumNeurons(); bCount++) {
                    saveFile << "," << curBiases[bCount];
                }
                saveFile << endl;
            }
        }
        saveFile.close();
    }
    catch (...) {
        return false;
    }
    return true;
}

void ConvNet::backPropagate(vector<tensor> trainData, vector<int> trainLabels, int iterations) {
    *this->progressGoal = iterations * trainData.size();
    // set pre kernel delta size
    vector<double> preKernDeltas(convLayers[convLayers.size() - 1]->getNumKernels() * pow(convLayers[convLayers.size() - 1]->getOutputDim(), 2));
    // one-hot encode all labels
    auto labels = oneHotEncode(std::move(trainLabels));
    // for every iteration
    for (int iters = 0; iters < iterations; iters++) {
        // for every data point
        for (int dIndex = 0; dIndex < trainData.size(); dIndex++) {
            *this->curProgress += 1;
            vector<double> curLabel = labels[dIndex];
            // calculate network output
            vector<double> result = propagate(trainData[dIndex]);
            // calculate output deltas
            vector<double> flatDeltas = denseLayers[denseLayers.size()-1]->backPropagate(curLabel);

            // calculate hidden deltas
            for (int layerIndex = denseLayers.size() - 2; layerIndex >= 0; layerIndex--) {
                flatDeltas = denseLayers[layerIndex]->backPropagate(flatDeltas, denseLayers[layerIndex + 1]->getWeights());
            }
            // convert flat deltas to kernel shaped deltas
            matrix lastFlatWeights;
            lastFlatWeights = denseLayers[0]->getWeights();
            for (int k = 0; k < preKernDeltas.size(); k++) {
                double total = 0;
                for (int d = 0; d < flatDeltas.size(); d++) {
                    total += flatDeltas[d] * lastFlatWeights[d][k];
                }
                preKernDeltas[k] = total;
            }
            tensor kernDeltas = reshapeVector(preKernDeltas, convLayers[convLayers.size() - 1]->getNumKernels(), convLayers[convLayers.size() - 1]->getOutputDim(), convLayers[convLayers.size() - 1]->getOutputDim());
            // calculate kernel deltas
            for (int kernelIndex = convLayers.size() - 1; kernelIndex >= 0; kernelIndex--) {
                if (kernelIndex != convLayers.size() - 1) {
                    kernDeltas = delPadding(kernDeltas, convLayers[kernelIndex + 1]->getPadWidth());
                }
                kernDeltas = convLayers[kernelIndex]->backPropagate(kernDeltas);
            }
            // update layer parameters
            for (auto & convLayer : convLayers) {
                convLayer->update();
            }
            for (auto & denseLayer  : denseLayers) {
                denseLayer->update();
            }
        }
    }
    *doneTraining = true;
}

void ConvNet::miniBatchBackPropagate(vector<tensor> trainData, vector<int> trainLabels, int iterations, int batchSize) {
    *this->progressGoal = iterations * trainData.size();
    // set pre kernel delta size
    vector<double> preKernDeltas(convLayers[convLayers.size() - 1]->getNumKernels() * pow(convLayers[convLayers.size() - 1]->getOutputDim(), 2));
    // one-hot encode all labels
    auto labels = oneHotEncode(std::move(trainLabels));
    // for every iteration
    for (int iters = 0; iters < iterations; iters++) {
        // randomize data before creating batches
        vector<int> indexes(trainData.size());
        for (int i = 0; i < trainData.size(); ++i)
            indexes[i] = i;

        std::random_shuffle(indexes.begin(), indexes.end());
        vector<tensor> newData(trainData.size());
        vector<vector<double>> newExpect(trainData.size());
        for (unsigned int i = 0; i < trainData.size(); i++) {
            newData[i] = trainData[indexes[i]];
            newExpect[i] = labels[indexes[i]];
        }
        trainData = newData;
        labels = newExpect;

        // get number of mini-batches
        int numBatches = floor(trainData.size() / batchSize);
        // allocate batch vector
        vector<vector<tensor>> allBatchData(numBatches);
        vector<vector<vector<double>>> allBatchLabels(numBatches);
        // create mini-batches
        for (int b = 0; b < numBatches; b++) {
            vector<tensor> curBatchData(batchSize);
            vector<vector<double>> curBatchLabels(batchSize);
            for (int i = 0; i < batchSize; i++) {
                curBatchData[i] = trainData[b * batchSize + i];
                curBatchLabels[i] = labels[b * batchSize + i];
            }
            allBatchData[b] = curBatchData;
            allBatchLabels[b] = curBatchLabels;
        }
        // for every mini-batch
        for (int bIndex = 0; bIndex < numBatches; bIndex++) {
            // for every data point
            for (int dIndex = 0; dIndex < batchSize; dIndex++) {
                *this->curProgress += 1;
                vector<double> curLabel = allBatchLabels[bIndex][dIndex];
                // calculate network output
                vector<double> result = propagate(allBatchData[bIndex][dIndex]);
                // calculate output deltas
                vector<double> flatDeltas = denseLayers[denseLayers.size()-1]->miniBatchBackPropagate(curLabel, batchSize);

                // calculate hidden deltas
                for (int layerIndex = denseLayers.size() - 2; layerIndex >= 0; layerIndex--) {
                    flatDeltas = denseLayers[layerIndex]->miniBatchBackPropagate(flatDeltas, batchSize, denseLayers[layerIndex +1]->getWeights());
                }
                // convert flat deltas to kernel shaped deltas
                matrix lastFlatWeights;
                lastFlatWeights = denseLayers[0]->getWeights();
                for (int k = 0; k < preKernDeltas.size(); k++) {
                    double total = 0;
                    for (int d = 0; d < flatDeltas.size(); d++) {
                        total += flatDeltas[d] * lastFlatWeights[d][k];
                    }
                    preKernDeltas[k] = total;
                }
                tensor kernDeltas = reshapeVector(preKernDeltas, convLayers[convLayers.size() - 1]->getNumKernels(),
                                                  convLayers[convLayers.size() - 1]->getOutputDim(),
                                                  convLayers[convLayers.size() - 1]->getOutputDim());
                // calculate kernel deltas
                for (int convIndex = convLayers.size() - 1; convIndex >= 0; convIndex--) {
                    if (convIndex != convLayers.size() - 1) {
                        kernDeltas = delPadding(kernDeltas, convLayers[convIndex + 1]->getPadWidth());
                    }
                    kernDeltas = convLayers[convIndex]->miniBatchBackPropagate(kernDeltas, batchSize);
                }
            }
            // update layer parameters
            for (auto &convLayer : convLayers) {
                convLayer->update();
                convLayer->resetDeltas();
            }
            for (auto &denseLayer : denseLayers) {
                denseLayer->update();
                denseLayer->resetDeltas();
            }
        }
    }
    *doneTraining = true;
}

void ConvNet::progressBar() {
    int barWidth = this->barSize;
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_CURSOR_INFO cursorInfo;
    GetConsoleCursorInfo(out, &cursorInfo);
    cursorInfo.bVisible = false;
    SetConsoleCursorInfo(out, &cursorInfo);
    StopWatch progressWatch;
    progressWatch.reset();
    while (!*doneTraining) {
        printBar(*curProgress, *progressGoal, barWidth, progressWatch);
    }
    printBar(1, 1, barWidth, progressWatch);
    cout << endl;
    cursorInfo.bVisible = true;
    SetConsoleCursorInfo(out, &cursorInfo);
}

void ConvNet::printBar(int curVal, int goal, int barWidth, StopWatch watch) {
    double progress = double(curVal) / goal;
    cout << "\r[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << std::setw(3) << int(progress * 100.0) << "% " << loading[int(watch.elapsed_time() / 100) % 4];
    cout.flush();
}

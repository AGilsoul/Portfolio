#include <iostream>
#include <ostream>
#include <sstream>
#include <iterator>
#include "../include/ConvNet.h"

using std::cin;
using std::ifstream;
using std::ios;

void readMnistFile(vector<tensor>& testData, vector<int>& expected, int numChannels=1, int numDuplicationChannels=1);

int main() {

    string placeHolder;
    int actualDataChannels = 1;
    int numDupChannels = 1;

    //ConvNet cnn(numDupChannels, 28, 0.001, 0.9);
    //cnn.addConvLayer(9, 3, true, true, "max");
    //cnn.addConvLayer(18, 3, true, true, "max");
    //cnn.addDenseLayer(128,true);
    //cnn.addDenseLayer(10, false);

    cout << "Loading Network..." << endl;
    ConvNet cnn("../res/CNNModel.csv");
    cnn.printNet();
    cout << endl;

    vector<tensor> data;
    vector<int> labels;
    cout << "Reading File..." << endl;
    readMnistFile(data, labels, actualDataChannels, numDupChannels);

    cout << "Randomizing Data..." << endl;
    cnn.shuffleData(data, labels);

    cout << "Normalizing..." << endl;
    normalize(data, {{0,255}});
    vector<tensor> trainData(data.begin(), data.begin() + 40000);
    vector<int> trainLabels(labels.begin(), labels.begin() + 40000);
    vector<tensor> testData(data.begin() + 40000, data.end());
    vector<int> testLabels(labels.begin() + 40000, labels.end());
    //cout << "Fitting Model..." << endl;
    //cnn.fitModel(trainData, trainLabels, 5, 1, true);
    cout << endl << "Testing Model..." << endl;
    auto res = cnn.eval(testData, testLabels) * 100;
    cout << endl << "Test Accuracy: " << setprecision(4) << res << "%" << endl;

    /*
    cout << endl << "Saving..." << endl;
    bool saved = cnn.save("CNNModel.csv");
    if (saved) {
        cout << "Save Successful" << endl;
    }
    else {
        cout << "Error While Saving" << endl;
    }
    */

    cout << "Press x to continue" << endl;
    cin >> placeHolder;
    return 0;
}


void readMnistFile(vector<tensor>& testData, vector<int>& expected, int numChannels, int numDuplicationChannels) {
    //784 data columns, one label column at beginning
    //strings to be used for reference and assignment of values when reading the file and assigning to the string list sList
    string sList[785];
    for (unsigned int i = 0; i < 785; i++) {
        string temp = "";
        sList[i] = temp;
    }
    //Reads from the file "mnist_train.csv"
    ifstream fin("../res/mnist_train.csv", ios::in);
    //vector<string> labels;
    int listSize = sizeof(sList) / sizeof(sList[0]);
    while (!fin.eof()) {
        vector<double> dData;
        int result;
        for (unsigned int i = 0; i < listSize; i++) {
            if (i != listSize - 1) {
                getline(fin, sList[i], ',');
            }
            else {
                getline(fin, sList[i], '\n');
            }

            if (i != 0) {
                dData.push_back(stod(sList[i]));
            }
            else {
                result = stoi(sList[i]);
            }
        }
        vector<double> finData;
        for (int chan = 0; chan < numDuplicationChannels; chan++) {
            finData.insert(finData.end(), dData.begin(), dData.end());
        }
        expected.push_back({result});
        testData.push_back(reshapeVector(finData, numChannels * numDuplicationChannels, 28, 28));
    }
    fin.close();
}

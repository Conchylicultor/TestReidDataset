#include <iostream>

#include "dataset.h"

using namespace std;



int main()
{
    cout << "Loading dataset..." << endl;

    Dataset dataset("/home/etienne/__A__/Data/Dataset/Set_1/");

    cout << "Sample selection..." << endl;

    // Choose randomly the training/testing sets
    dataset.selectPairs();

    cout << "Features extractions..." << endl;

    // Extract features vector for each pairs
    dataset.computeFeatures();

    cout << "Training..." << endl;

    // Machine learing algorithm
    dataset.train();

    cout << "Testing..." << endl;

    // Test and plot results
    dataset.test();

    return 0;
}

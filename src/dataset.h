#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class Dataset
{
public:
    Dataset(string folderUrl);

    // Choose randomly the training/testing sets
    void selectPairs();

    // Extract features vector for each pairs
    void computeFeatures();

    // Machine learing algorithm
    void train();

    // Test and plot results
    void test();
};

#endif // DATASET_H

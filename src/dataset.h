#ifndef DATASET_H
#define DATASET_H

#include "person.h"

#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

struct PairSample
{
    string first;
    string second;
    bool samePerson;
};

class Dataset
{
public:
    // Loading persons
    Dataset(string folderUrl_);

    // Choose randomly the training/testing sets
    void selectPairs();

    // Extract features vector for each pairs
    void computeFeatures();

    // Machine learing algorithm
    void train();

    // Test and plot results
    void test();

private:
    string folderUrl;
    vector<Person> listPersons;

    // TODO: Pre-allocation
    vector<PairSample> listSamples;// TODO: Add more informations (which cam,...), the informations can be found on a file personId.txt

    void histRGB(const Mat &frame, const Mat &fgMask, array<Mat, 3> &histogramChannels);

    cv::Mat trainingData;
    cv::Mat trainingClasses;

    cv::Mat testData;
};

#endif // DATASET_H

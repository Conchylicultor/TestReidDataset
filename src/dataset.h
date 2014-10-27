#ifndef DATASET_H
#define DATASET_H

#include "person.h"

#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

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

    list<pair<string, string> > positiveSamples;
    list<pair<string, string> > negativeSamples;// TODO: Add more informations (which cam,...), the informations can be found on a file personId.txt
};

#endif // DATASET_H

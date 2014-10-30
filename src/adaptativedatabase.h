#ifndef ADAPTATIVEDATABASE_H
#define ADAPTATIVEDATABASE_H

#include <string>
#include <array>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

struct FeaturesElement
{
    array<Mat, 3> histogramChannels;
};

struct PersonElement
{
    vector<FeaturesElement> features;
    float thresholdValue;

    vector<string> sampleImages; // For testing and debuging, to plot the person
};

class AdaptativeDatabase
{
public:
    AdaptativeDatabase(string folderUrl_);

    void main();

private:
    string folderUrl;

    vector<vector<string>> listSequence;

    vector<PersonElement> listDatabase;

    // Compute features
    void histRGB(const Mat &frame, const Mat &fgMask, array<Mat, 3> &histogramChannels);

    void loadMachineLearning();
    CvSVM svm;
};

#endif // ADAPTATIVEDATABASE_H

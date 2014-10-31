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

    string name;
};

struct SequenceElement
{
    vector<string> listFrameIds;
    string name;
};

class AdaptativeDatabase
{
public:
    AdaptativeDatabase(string folderUrl_);

    void main();

private:
    string folderUrl;

    vector<SequenceElement> listSequence;

    vector<PersonElement> listDatabase;

    // Compute features
    void histRGB(const Mat &frame, const Mat &fgMask, array<Mat, 3> &histogramChannels);

    void loadMachineLearning();
    CvSVM svm;

    float distance(const FeaturesElement &elem1, const FeaturesElement &elem2);

    void debugShowImgs(const vector<string> &idsImgs, int nbPos);

};

#endif // ADAPTATIVEDATABASE_H

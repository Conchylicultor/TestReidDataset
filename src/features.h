#ifndef FEATURES_H
#define FEATURES_H


#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

// Independent features

#define NB_MAJOR_COLORS 3

struct MajorColorElem
{
    Scalar color;
    float position;
    int weightColor; // Nb of element of this color
};

// Global structs

struct FeaturesElement
{
    array<Mat, 3> histogramChannels;
    array<MajorColorElem, NB_MAJOR_COLORS> majorColors;
};

class Features
{
public:
    static void computeFeature(const string &id, FeaturesElement &featuresElemOut);
    static void computeDistance(const FeaturesElement &featuresElemIn1, const FeaturesElement &featuresElemIn2, Mat rowFeatureVector);
private:
    static void histRGB(const Mat &frame, const Mat &fgMask, array<Mat, 3> &histogramChannels);
    static void majorColors(const Mat &frame, const Mat &fgMask, array<MajorColorElem, NB_MAJOR_COLORS> &listMajorColors);

};

#endif // FEATURES_H

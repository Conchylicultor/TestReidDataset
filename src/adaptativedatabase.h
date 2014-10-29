#ifndef ADAPTATIVEDATABASE_H
#define ADAPTATIVEDATABASE_H

#include <string>
#include <array>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

struct featuresElement
{
    array<Mat, 3> histogramChannels;
};

class AdaptativeDatabase
{
public:
    AdaptativeDatabase(string folderUrl_);

    void main();

private:
    string folderUrl;

    vector<vector<string>> listSequence;

    // Compute features
    void histRGB(const Mat &frame, const Mat &fgMask, array<Mat, 3> &histogramChannels);
};

#endif // ADAPTATIVEDATABASE_H

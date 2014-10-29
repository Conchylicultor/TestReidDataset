#ifndef ADAPTATIVEDATABASE_H
#define ADAPTATIVEDATABASE_H

#include <string>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class AdaptativeDatabase
{
public:
    AdaptativeDatabase(string folderUrl_);

    void main();

private:
    string folderUrl;

    vector<vector<string>> listSequence;
};

#endif // ADAPTATIVEDATABASE_H

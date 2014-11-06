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

struct EvaluationElement
{
    // X Datas
    int nbSequence;

    // Y Datas
    int nbError;
    int nbSuccess; // Final result

    int nbErrorFalsePositiv;// << Database corrupted
    int nbErrorFalseNegativ;

    int nbErrorWithoutClone;// Errors exept if match at least recognize (at least) by one of it's clone in the dataset
    int nbErrorPersonAdded;// When some is added but is already in the datset
    int nbClone;// When some is added but is already in the datset

    int nbPersonAdded;// Infos
};

class AdaptativeDatabase
{
public:
    AdaptativeDatabase(string folderUrl_);

    void main();
    void plotEvaluation();

private:
    string folderUrl;

    vector<SequenceElement> listSequence;

    vector<PersonElement> listDatabase;

    // Compute features
    void histRGB(const Mat &frame, const Mat &fgMask, array<Mat, 3> &histogramChannels);

    void loadMachineLearning();
    CvSVM svm;

    float distance(const FeaturesElement &elem1, const FeaturesElement &elem2);

    // Debug and evaluations
    void debugShowImgs(const vector<string> &idsImgs, int nbPos);
    vector<EvaluationElement> listEvaluation;// Evaluation which contain the datas to plot
};

#endif // ADAPTATIVEDATABASE_H

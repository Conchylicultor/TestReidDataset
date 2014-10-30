#include "adaptativedatabase.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>

#define HIST_SIZE 100

AdaptativeDatabase::AdaptativeDatabase(string folderUrl_) :
    folderUrl(folderUrl_)
{
    // Initialize the randomness
    std::srand ( unsigned ( std::time(0) ) );

    ifstream fileListPersons(folderUrl + "traces.txt");
    if(!fileListPersons.is_open())
    {
        cout << "Error: Impossible to load the database file: "<< folderUrl << endl;
        exit(0);
    }

    // /!\ Warning: No verification on the file structure (can be corrupted)

    int lastSequenceNumber(0);
    for(string line; std::getline(fileListPersons, line); )
    {
        // If new group of image
        if(line.find("-----") != std::string::npos)
        {
            // Ignoring the line
        }
        // Otherwise, match the sequence
        else
        {
            // Extract sequence id number
            string seqIdString;
            istringstream lineStream(line);
            std::getline(lineStream, seqIdString, '_' );
            int seqIdNum = stoi(seqIdString);

            if(seqIdNum > lastSequenceNumber)// New sequence
            {
                listSequence.push_back(vector<string>());
                lastSequenceNumber = seqIdNum;
            }
            listSequence.back().push_back(line);
        }
    }

    fileListPersons.close();

    // Load SVM from file
    loadMachineLearning();
}

void AdaptativeDatabase::main()
{
    // Process:
    std::random_shuffle(listSequence.begin(), listSequence.end());

    for(vector<string> currentSequence : listSequence)
    {
        // Read/load the new sequence

        // TODO: Selection only some images and not the complete sequence
        vector<FeaturesElement> listSequence;
        listSequence.reserve(currentSequence.size());

        for(string currentIdString : currentSequence)
        {
            Mat img = imread(folderUrl + currentIdString + ".png");
            Mat mask = imread(folderUrl + currentIdString + "_mask.png");

            if (img.empty() || mask.empty())
            {
                cout << "Error: cannot loading images (id=" << currentIdString << ")" << endl;
                continue;
            }

            cvtColor(mask, mask,CV_BGR2GRAY);
            threshold(mask, mask, 254, 255, THRESH_BINARY);// Convert to binary


            listSequence.push_back(FeaturesElement());

            histRGB(img, mask, listSequence.back().histogramChannels);
        }

        bool newPers(true);

        // Select persons on the database and compute distance
        for(PersonElement currentPerson : listDatabase)
        {
        }

        if(newPers)
        {
            // Add the new person to the database
            listDatabase.push_back(PersonElement());
            listDatabase.back().features.swap(listSequence);
            listDatabase.back().sampleImages = currentSequence;

            // Compute the threshold value
        }
        else
        {
            // Update the match
        }
        // TODO: Manualy label the person if wrong
    }
}

void AdaptativeDatabase::histRGB(const Mat &frame, const Mat &fgMask, array<Mat, 3> &histogramChannels)
{
    // Conversion to the right color space ???
    // Size of the histogram
    int histSize = HIST_SIZE; // bin size
    float range[] = {0, 256}; // min max values
    const float *ranges[] = {range};
    // Extraction of the histograms
    std::vector<cv::Mat> sourceChannels;
    cv::split(frame, sourceChannels);
    cv::calcHist(&sourceChannels[0], 1, 0, fgMask, histogramChannels[0], 1, &histSize, ranges, true, false );
    cv::calcHist(&sourceChannels[1], 1, 0, fgMask, histogramChannels[1], 1, &histSize, ranges, true, false );
    cv::calcHist(&sourceChannels[2], 1, 0, fgMask, histogramChannels[2], 1, &histSize, ranges, true, false );
    // Normalize
    normalize(histogramChannels[0], histogramChannels[0]);
    normalize(histogramChannels[1], histogramChannels[1]);
    normalize(histogramChannels[2], histogramChannels[2]);
}

void AdaptativeDatabase::loadMachineLearning()
{
    // Loading file
    FileStorage fileTraining(folderUrl + "training.yml", FileStorage::READ);

    if(!fileTraining.isOpened())
    {
        cout << "Error: cannot open the training file " << folderUrl + "training.yml" << endl;
        exit(0);
    }

    Mat trainingData;
    Mat trainingClasses;
    fileTraining["trainingData"] >> trainingData;
    fileTraining["trainingClasses"] >> trainingClasses;

    fileTraining.release();

    // Training
    CvSVMParams param = CvSVMParams();

    param.svm_type = CvSVM::C_SVC;
    param.kernel_type = CvSVM::RBF; //CvSVM::RBF, CvSVM::LINEAR ...
    param.degree = 0; // for poly
    param.gamma = 20; // for poly/rbf/sigmoid
    param.coef0 = 0; // for poly/sigmoid

    param.C = 7; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
    param.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
    param.p = 0.0; // for CV_SVM_EPS_SVR

    param.class_weights = NULL; // for CV_SVM_C_SVC
    param.term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
    param.term_crit.max_iter = 1000;
    param.term_crit.epsilon = 1e-6;
 
    svm.train_auto(trainingData, trainingClasses, cv::Mat(), cv::Mat(), param);

    cout << "Training complete." << endl;
}

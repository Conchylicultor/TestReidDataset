#include "adaptativedatabase.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

#include "utils.h"

#define HIST_SIZE 100
static const size_t minSequenceSize = 5;
static const int nbInitialisationPairs = 10;
static const int nbTestingPairs = 10;

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

    string currentPersName;

    int lastSequenceNumber(0);
    for(string line; std::getline(fileListPersons, line); )
    {
        // If new group of image
        if(line.find("-----") != std::string::npos)
        {
            utils::replace(line, "----- ", "");
            utils::replace(line, " -----", "");
            currentPersName = line;
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
                listSequence.push_back(SequenceElement());
                listSequence.back().name = currentPersName;
                lastSequenceNumber = seqIdNum;
            }
            listSequence.back().listFrameIds.push_back(line);
            // TODO: Add also the real result: listIdentity.back().push_back(namePerson)
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

    listEvaluation.push_back(EvaluationElement{0,0,0,0,0,0,0,0,0});

    for(SequenceElement currentSequence : listSequence)
    {
        cout << "--------------------------------------" << endl;
        if(currentSequence.listFrameIds.size() < minSequenceSize)
        {
            cout << "Pb: Not enougth pairs for the current sequence. Cannot add it " << currentSequence.listFrameIds.at(0) << endl;
            continue;
        }

        // Evaluation which contain the datas to plot
        EvaluationElement newEvalElement;
        // X
        newEvalElement.nbSequence = listEvaluation.size() + 1;
        // Y: Cumulative results
        newEvalElement.nbError             = listEvaluation.back().nbError;
        newEvalElement.nbSuccess           = listEvaluation.back().nbSuccess;
        newEvalElement.nbErrorFalsePositiv = listEvaluation.back().nbErrorFalsePositiv;
        newEvalElement.nbErrorFalseNegativ = listEvaluation.back().nbErrorFalseNegativ;
        newEvalElement.nbErrorPersonAdded  = listEvaluation.back().nbErrorPersonAdded;
        newEvalElement.nbErrorWithoutClone = listEvaluation.back().nbErrorWithoutClone;
        newEvalElement.nbClone             = listEvaluation.back().nbClone;
        newEvalElement.nbPersonAdded       = listEvaluation.back().nbPersonAdded;
        listEvaluation.push_back(newEvalElement);

        // Read/load the new sequence

        // TODO: Selection only some images and not the complete sequence
        vector<FeaturesElement> listSequenceFeatures;
        listSequenceFeatures.reserve(currentSequence.listFrameIds.size());

        for(string currentIdString : currentSequence.listFrameIds)
        {
            listSequenceFeatures.push_back(FeaturesElement());
            Features::computeFeature(folderUrl + currentIdString, listSequenceFeatures.back());
        }

        bool newPers(true);
        bool alreadyInDataset(false);
        bool isRecognizeOnce(false);
        int nbErrorClone(0);
        for(PersonElement currentPerson : listDatabase)
        {
            if(currentPerson.name == currentSequence.name)
            {
                alreadyInDataset = true;
            }
        }

        // Select persons on the database and compute distance
        for(PersonElement currentPerson : listDatabase)
        {
            float thresholdValue = 0.0;

            // Randomly compare persons in the dataset
            for(int i = 0 ; i < nbTestingPairs ; ++i)
            {
                int number1 = std::rand() % listSequenceFeatures.size();
                int number2 = std::rand() % currentPerson.features.size();

                if(number1 != number2)
                {
                    Mat rowFeatureVector;

                    Features::computeDistance(listSequenceFeatures.at(number1),
                                              currentPerson.features.at(number2),
                                              rowFeatureVector);

                    thresholdValue += svm.predict(rowFeatureVector);
                }
                else
                {
                    --i;
                }
            }

            thresholdValue /= nbTestingPairs;

            bool isErrorDebug = false;

            // Depending of the thresholdValue, reid or not
            if(thresholdValue > 0.5)
            {
                cout << "Match (" << thresholdValue << ") : " << currentPerson.name;
                newPers = false;

                if (currentPerson.name != currentSequence.name) // False positiv
                {
                    cout << " <<< ERROR";

                    listEvaluation.back().nbError++;
                    listEvaluation.back().nbErrorFalsePositiv++;
                    listEvaluation.back().nbErrorWithoutClone++;

                    isErrorDebug = true;
                }
                else
                {
                    isRecognizeOnce = true;
                }

                cout << endl;
            }
            else
            {
                cout << "Diff (" << thresholdValue << ")";

                if (currentPerson.name == currentSequence.name) // False negativ
                {
                    cout << " <<< ERROR";

                    listEvaluation.back().nbError++;
                    listEvaluation.back().nbErrorFalseNegativ++;
                    nbErrorClone++;

                    isErrorDebug = true;
                }
                cout << endl;
            }

            /*if(isErrorDebug)
            {
                debugShowImgs(currentSequence.listFrameIds, 0);
                debugShowImgs(currentPerson.sampleImages, 1);
                cv::waitKey(0);
            }*/
        }

        if(alreadyInDataset && !isRecognizeOnce)
        {
            listEvaluation.back().nbErrorWithoutClone += nbErrorClone;
        }

        if(alreadyInDataset && newPers)
        {
            listEvaluation.back().nbClone++;
        }

        if(newPers)
        {
            // Compute the threshold value
            float thresholdValue = 0.0;

            cout << "No match: Add the new person to the dataset : " << currentSequence.name << endl;

            // Add the new person to the database
            listDatabase.push_back(PersonElement());
            listDatabase.back().features.swap(listSequenceFeatures);
            listDatabase.back().sampleImages = currentSequence.listFrameIds;
            listDatabase.back().thresholdValue = thresholdValue;
            listDatabase.back().name = currentSequence.name;

            listEvaluation.back().nbPersonAdded++;
        }
        else
        {
            // TODO: Update the match
        }
        // TODO: Manualy label the person if wrong
    }
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

void AdaptativeDatabase::debugShowImgs(const vector<string> &idsImgs, int nbPos)
{
    int number1 = std::rand() % idsImgs.size();
    int number2 = std::rand() % idsImgs.size();
    int number3 = std::rand() % idsImgs.size();

    Mat img1 = imread(folderUrl + idsImgs.at(number1) + ".png");
    Mat img2 = imread(folderUrl + idsImgs.at(number2) + ".png");
    Mat img3 = imread(folderUrl + idsImgs.at(number3) + ".png");

    if (img1.empty() || img2.empty() || img3.empty())
    {
        cout << "Error loading images" << endl;
        return;
    }

    //show the image
    imshow(std::to_string(nbPos) + " - Img1", img1);
    imshow(std::to_string(nbPos) + " - Img2", img2);
    imshow(std::to_string(nbPos) + " - Img3", img3);

    moveWindow(std::to_string(nbPos) + " - Img1", 0*200 , nbPos * 200);
    moveWindow(std::to_string(nbPos) + " - Img2", 1*200 , nbPos * 200);
    moveWindow(std::to_string(nbPos) + " - Img3", 2*200 , nbPos * 200);
}

void AdaptativeDatabase::plotEvaluation()
{
    const int stepHorizontalAxis = 20;
    const int stepVerticalAxis = 20;
    const int windowsEvalHeight = 900;

    Mat imgEval(Size(stepHorizontalAxis * listEvaluation.size(), windowsEvalHeight),
                CV_8UC3,
                Scalar(0,0,0));

    for(size_t i = 1 ; i < listEvaluation.size() ; ++i)
    {
        EvaluationElement evalElemPrev = listEvaluation.at(i-1);
        EvaluationElement evalElemNext = listEvaluation.at(i);

        Point pt1;
        Point pt2;
        pt1.x = stepHorizontalAxis * evalElemPrev.nbSequence;
        pt2.x = stepHorizontalAxis * evalElemNext.nbSequence;

        Scalar color;

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbPersonAdded;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbPersonAdded;
        color = Scalar(255, 0, 0);
        putText(imgEval, "Person added", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbError;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbError;
        color = Scalar(0, 255, 0);
        putText(imgEval, "Errors (Cumulativ)", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbErrorFalseNegativ;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbErrorFalseNegativ;
        color = Scalar(0, 255, 255);
        putText(imgEval, "False negativ", Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbErrorFalsePositiv;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbErrorFalsePositiv;
        color = Scalar(0, 130, 255);
        putText(imgEval, "False positiv", Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbErrorWithoutClone;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbErrorWithoutClone;
        color = Scalar(115, 32, 150);
        putText(imgEval, "Without clone", Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbClone;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbClone;
        color = Scalar(73, 92, 17);
        putText(imgEval, "Clones", Point(10, 70), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);

        pt1.y = windowsEvalHeight - stepVerticalAxis * (evalElemNext.nbError - evalElemPrev.nbError);
        pt2.y = windowsEvalHeight - stepVerticalAxis * (evalElemNext.nbError - evalElemPrev.nbError);
        color = Scalar(255, 255, 0);
        putText(imgEval, "Errors", Point(10, 10), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);
    }

    // Display
    namedWindow("Evaluation Results", CV_WINDOW_AUTOSIZE);
    imshow("Evaluation Results", imgEval);

    waitKey();
}

#include "dataset.h"

#include <fstream>
#include <regex>
#include <array>
#include <algorithm>
#include <cstdlib>

#include "utils.h"

#define NB_SELECTED_PAIR 10

#define HIST_SIZE 100

#define RECORD_SVM_TRAINING 0


Dataset::Dataset(string folderUrl_) :
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

    // /!\ Warning: No verification on the file

    Person *currentPers = nullptr;
    for(string line; std::getline(fileListPersons, line); )
    {
        // If new group of image
        if(line.find("-----") != std::string::npos)
        {
            utils::replace(line, "----- ", "");
            utils::replace(line, " -----", "");

            // Check if already on the list
            bool onTheList = false;
            for(Person &iter : listPersons)
            {
                if(iter.getName() == line)
                {
                    currentPers = &iter;
                    onTheList = true;
                }
            }

            // New element
            if(!onTheList)
            {
                listPersons.push_back(Person(line));
                currentPers = &listPersons.back();
            }
        }
        // Otherwise, simply add the image to the current person
        else
        {
            currentPers->addImageId(line);
        }
    }

    fileListPersons.close();
}

void Dataset::selectPairs()
{
    // TODO: Divide the listPersons between training and testing set

    // Training pairs
    std::random_shuffle(listPersons.begin(), listPersons.end());

    // Positives sample
    for(Person &iter : listPersons) // For each person...
    {
        // ... select some random pairs
        for(int i = 0 ; i < NB_SELECTED_PAIR ; ++i) // TODO ?: Replace the arbitrary choosen number ?
        {
            if(iter.getListImagesId().size() == 1)
            {
                cout << "Error: Sequence of one image" << endl;
                break;
            }

            int number1 = std::rand() % iter.getListImagesId().size();
            int number2 = std::rand() % iter.getListImagesId().size();

            // TODO: Check that the couple has not been selected yet
            if(number1 != number2)
            {
                listSamples.push_back(PairSample{iter.getListImagesId().at(number1),
                                                 iter.getListImagesId().at(number2),
                                                 true});
            }
            else
            {
                --i; // Numbers are equal, does not count
            }
        }
    }

    // Negative sample
    for(size_t i = 0 ; i < listPersons.size() * 2 ; ++i)
    {
        // Selection of two different persons
        int numberPers1 = std::rand() % listPersons.size();
        int numberPers2 = std::rand() % listPersons.size();

        if(numberPers1 != numberPers2)
        {
            for(int j = 0 ; j < NB_SELECTED_PAIR/2 ; ++j)
            {
                int number1 = std::rand() % listPersons.at(numberPers1).getListImagesId().size();
                int number2 = std::rand() % listPersons.at(numberPers2).getListImagesId().size();

                // TODO: Check that the couple has not been selected yet
                listSamples.push_back(PairSample{listPersons.at(numberPers1).getListImagesId().at(number1),
                                                 listPersons.at(numberPers2).getListImagesId().at(number2),
                                                 false});
            }
        }
        else
        {
            --i; // Numbers are equal, does not count
        }
    }

    // Testing pairs
}

void Dataset::computeFeatures()
{
    std::random_shuffle(listSamples.begin(), listSamples.end());

    //trainingData.reserve();
    //trainingClasses;

    //for(PairSample iter : listSamples)
    //{
    for(size_t i = 0 ; i < listSamples.size() ; ++i)
    {
        PairSample &iter = listSamples.at(i);

        // Read images
        Mat imgPers1     = imread(folderUrl + iter.first + ".png");
        Mat imgMaskPers1 = imread(folderUrl + iter.first + "_mask.png");

        Mat imgPers2     = imread(folderUrl + iter.second + ".png");
        Mat imgMaskPers2 = imread(folderUrl + iter.second + "_mask.png");


        if (imgPers1.empty() || imgPers2.empty() || imgMaskPers1.empty() || imgMaskPers2.empty())
        {
            cout << "Error: cannot loading images" << endl;
            cout << "    " << iter.first << endl;
            cout << "    " << iter.second << endl;
            continue;
        }

        cvtColor(imgMaskPers1,imgMaskPers1,CV_BGR2GRAY);
        threshold(imgMaskPers1, imgMaskPers1, 254, 255, THRESH_BINARY);

        cvtColor(imgMaskPers2,imgMaskPers2,CV_BGR2GRAY);
        threshold(imgMaskPers2, imgMaskPers2, 254, 255, THRESH_BINARY);

        // Compute features for each person

        array<Mat, 3> histogramChannelsPers1;
        array<Mat, 3> histogramChannelsPers2;
        histRGB(imgPers1, imgMaskPers1, histogramChannelsPers1);
        histRGB(imgPers2, imgMaskPers2, histogramChannelsPers2);

        // Compute distance and add feature vector to the training set
        Mat rowFeatureVector = cv::Mat::ones(1, 3, CV_32FC1);

        rowFeatureVector.at<float>(0,0) = compareHist(histogramChannelsPers1.at(0), histogramChannelsPers2.at(0), CV_COMP_BHATTACHARYYA);
        rowFeatureVector.at<float>(0,1) = compareHist(histogramChannelsPers1.at(1), histogramChannelsPers2.at(1), CV_COMP_BHATTACHARYYA);
        rowFeatureVector.at<float>(0,2) = compareHist(histogramChannelsPers1.at(2), histogramChannelsPers2.at(2), CV_COMP_BHATTACHARYYA);

        array<MajorColorElem, NB_MAJOR_COLORS> majorColorsPers1;
        array<MajorColorElem, NB_MAJOR_COLORS> majorColorsPers2;

        majorColors(imgPers1, imgMaskPers1, majorColorsPers1);
        majorColors(imgPers2, imgMaskPers2, majorColorsPers2);

        // Distance for the major color
        float minDist;
        for(MajorColorElem currentElem1 : majorColorsPers1)
        {
            for(MajorColorElem currentElem2 : majorColorsPers2)
            {
                // dist = norm(currentElem1.color - currentElem2.color);
            }
        }

        // TODO: Add feature: camera id ; Add feature: time

        Mat rowClass = cv::Mat::ones(1, 1, CV_32FC1);
        if(iter.samePerson)
        {
            rowClass.at<float>(0,0) = 1;
        }
        else
        {
            rowClass.at<float>(0,0) = -1;
        }

        if(i > listSamples.size()/2)
        {
            trainingData.push_back(rowFeatureVector);
            trainingClasses.push_back(rowClass);
        }
        else
        {
            testData.push_back(rowFeatureVector);
        }
    }
}

void Dataset::train()
{
    trainSVM();
}

void Dataset::trainSVM()
{
    // TODO: Modify params
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

    if(RECORD_SVM_TRAINING)
    {
        FileStorage fileTraining(folderUrl + "training.yml", FileStorage::WRITE);

        fileTraining << "trainingData" << trainingData;
        fileTraining << "trainingClasses" << trainingClasses;

        fileTraining.release();

    }
}

void Dataset::test()
{
    float nbTrue = 0.0;
    float nbFalse = 0.0;
    for(int i = 0; i < testData.rows; i++)
    {
        cv::Mat sample = testData.row(i);

        float response = svm.predict(sample);

        if((response == 1  && listSamples.at(i).samePerson) ||
           (response == -1 && !listSamples.at(i).samePerson) )
        {
            nbTrue = nbTrue + 1.0;
        }
        else
        {
            nbFalse = nbFalse + 1.0;

            /*PairSample &iter = listSamples.at(i);

            Mat img1 = imread(folderUrl + iter.first + ".png");
            Mat img2 = imread(folderUrl + iter.second + ".png");

            //if fail to read the image
            if (img1.empty() || img2.empty())
            {
                cout << "Error loading images" << endl;
                exit(0);
            }

            //show the image
            cout << response << endl;
            imshow("Img1", img1);
            imshow("Img2", img2);

            // Wait until user press some key
            char key = waitKey(0);
            if(key == 32) // Spacebar
            {
                continue;
            }*/
        }
    }
    cout << "Results: " << nbTrue/(nbTrue+nbFalse)*100.0 << "%" << endl;
    cout << "F:" << nbFalse << " T:" << nbTrue << endl;
}


void Dataset::histRGB(const Mat &frame, const Mat &fgMask, array<Mat, 3> &histogramChannels)
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

void Dataset::majorColors(const Mat &frame, const Mat &fgMask, array<MajorColorElem, NB_MAJOR_COLORS> &listMajorColors)
{
    Mat src = frame.clone();

    // Step 1 : map the src to the samples
    Mat samples(cv::countNonZero(fgMask), 3, CV_32F); // We only cluster the "white" pixels

    int i = 0;
    for (int x = 0 ; x < fgMask.rows ; ++x)
    {
        for (int y = 0; y < fgMask.cols ; ++y)
        {
            if(fgMask.at<uchar>(x,y))
            {
                samples.at<float>(i,0) = src.at<Vec3b>(x,y)[0];
                samples.at<float>(i,1) = src.at<Vec3b>(x,y)[1];
                samples.at<float>(i,2) = src.at<Vec3b>(x,y)[2];
                ++i;
            }
        }
    }

    // Step 2 : apply kmeans to find labels and centers
    int clusterCount = NB_MAJOR_COLORS;
    cv::Mat labels;
    int attempts = 5;
    cv::Mat centers;
    cv::kmeans(samples, clusterCount, labels,
               cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                                10, 0.01),
               attempts, cv::KMEANS_PP_CENTERS, centers);

    // Step 3 : map the centers to the output
    i = 0;
    Mat dest(src.size(), src.type());
    for (int x = 0 ; x < dest.rows ; ++x)
    {
        for (int y = 0 ; y < dest.cols ; ++y)
        {
            if(fgMask.at<uchar>(x,y))
            {
                int cluster_idx = labels.at<int>(i,0);
                dest.at<Vec3b>(x,y)[0] = centers.at<float>(cluster_idx, 0);
                dest.at<Vec3b>(x,y)[1] = centers.at<float>(cluster_idx, 1);
                dest.at<Vec3b>(x,y)[2] = centers.at<float>(cluster_idx, 2);
                ++i;
            }
            else
            {
                dest.at<Vec3b>(x,y)[0] = 0;
                dest.at<Vec3b>(x,y)[1] = 0;
                dest.at<Vec3b>(x,y)[2] = 0;
            }
        }
    }

    // Step 4 : Fill information
    for(int i = 0 ; i < centers.rows ; ++i)
    {
        listMajorColors.at(i).color = Scalar(centers.at<float>(i, 0),
                                             centers.at<float>(i, 1),
                                             centers.at<float>(i, 2));

        // TODO: Add Spacial information

        // TODO: Add number of pixel of each major color
    }

    /*// Debug

    imshow("src", src);
    imshow("mask", fgMask);
    imshow("dest", dest);
    waitKey( 0 );*/
}

#include "features.h"

// Variables for features computation

#define HIST_SIZE 100

void Features::computeFeature(const string &id, FeaturesElement &featuresElemOut)
{
    // Read images
    Mat imgPers     = imread(id + ".png");
    Mat imgMaskPers = imread(id + "_mask.png");


    if (imgPers.empty() || imgMaskPers.empty())
    {
        cout << "Error: cannot loading images : " << endl;
        cout << id << endl;

        exit(0);
    }

    cvtColor(imgMaskPers, imgMaskPers, CV_BGR2GRAY);
    threshold(imgMaskPers, imgMaskPers, 254, 255, THRESH_BINARY);

    histRGB(imgPers, imgMaskPers, featuresElemOut.histogramChannels);
    majorColors(imgPers, imgMaskPers, featuresElemOut.majorColors);
}

void Features::computeDistance(const FeaturesElement &elem1, const FeaturesElement &elem2, Mat &rowFeatureVector)
{
    rowFeatureVector = cv::Mat::ones(1, 3 + NB_MAJOR_COLORS, CV_32FC1);

    // Histogram
    rowFeatureVector.at<float>(0,0) = compareHist(elem1.histogramChannels.at(0), elem2.histogramChannels.at(0), CV_COMP_BHATTACHARYYA);
    rowFeatureVector.at<float>(0,1) = compareHist(elem1.histogramChannels.at(1), elem2.histogramChannels.at(1), CV_COMP_BHATTACHARYYA);
    rowFeatureVector.at<float>(0,2) = compareHist(elem1.histogramChannels.at(2), elem2.histogramChannels.at(2), CV_COMP_BHATTACHARYYA);

    // Major colors
    int indexColumns = 3;
    for(MajorColorElem currentElem1 : elem1.majorColors)
    {
        float minDist = norm(currentElem1.color - elem2.majorColors.front().color);
        float dist = 0.0;
        for(MajorColorElem currentElem2 : elem2.majorColors)
        {
            dist = norm(currentElem1.color - currentElem2.color);
            if(dist < minDist)
            {
                minDist = dist;
            }
        }
        rowFeatureVector.at<float>(0,indexColumns) = minDist;
        indexColumns++;
    }

    // TODO: Add feature: camera id ; Add feature: time
}

void Features::histRGB(const Mat &frame, const Mat &fgMask, array<Mat, 3> &histogramChannels)
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

void Features::majorColors(const Mat &frame, const Mat &fgMask, array<MajorColorElem, NB_MAJOR_COLORS> &listMajorColors)
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

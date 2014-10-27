#include "dataset.h"

#include <fstream>
#include <regex>
#include <algorithm>
#include <cstdlib>

#define NB_SELECTED_PAIR 10

bool replace(std::string& str, const std::string& from, const std::string& to)
{
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}


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
            replace(line, "----- ", "");
            replace(line, " -----", "");

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
            int number1 = std::rand() % iter.getListImagesId().size();
            int number2 = std::rand() % iter.getListImagesId().size();

            // TODO: Check that the couple has not been selected yet
            if(number1 != number2)
            {
                positiveSamples.push_back(pair<string, string> (iter.getListImagesId().at(number1),
                                                                iter.getListImagesId().at(number2)));
            }
            else
            {
                --i; // Numbers are equal, does not count
            }
        }
    }

    // Negative sample
    for(int i = 0 ; i < listPersons.size() * 2 ; ++i)
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
                negativeSamples.push_back(pair<string, string> (listPersons.at(numberPers1).getListImagesId().at(number1),
                                                                listPersons.at(numberPers2).getListImagesId().at(number2)));
            }
        }
        else
        {
            --i; // Numbers are equal, does not count
        }
    }

    /*for (auto iter : positiveSamples)
    {
        cout << iter.first << " " << iter.second << endl;
    }
    cout << "iter.first <<  << iter.second" << endl;

    for (auto iter : negativeSamples)
    {
        cout << iter.first << " " << iter.second << endl;
    }

    for(auto iter : positiveSamples)
    {
        Mat img1 = imread(folderUrl + iter.first + ".png");
        Mat img2 = imread(folderUrl + iter.second + ".png");

        //if fail to read the image
        if (img1.empty() || img2.empty())
        {
            cout << "Error loading images" << endl;
            exit(0);
        }

        //show the image
        imshow("Img1", img1);
        imshow("Img2", img2);

        // Wait until user press some key
        char key = waitKey(0);
        if(key == 32) // Spacebar
        {
            continue;
        }
        else
        {
            break;
        }
    }

    for(auto iter : negativeSamples)
    {
        Mat img1 = imread(folderUrl + iter.first + ".png");
        Mat img2 = imread(folderUrl + iter.second + ".png");

        //if fail to read the image
        if (img1.empty() || img2.empty())
        {
            cout << "Error loading images" << endl;
            exit(0);
        }

        //show the image
        imshow("Img1", img1);
        imshow("Img2", img2);

        // Wait until user press some key
        char key = waitKey(0);
        if(key == 32) // Spacebar
        {
            continue;
        }
        else
        {
            break;
        }
    }*/

    // Testing pairs
}

void Dataset::computeFeatures()
{

}

void Dataset::train()
{

}

void Dataset::test()
{

}

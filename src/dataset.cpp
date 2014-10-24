#include "dataset.h"

#include <fstream>
#include <regex>
#include <algorithm>
#include <cstdlib>

bool replace(std::string& str, const std::string& from, const std::string& to)
{
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

template <class T>
void randomizeList(list<T> &sortedList)
{
    vector<T> tempVector{ std::make_move_iterator(std::begin(sortedList)),
                          std::make_move_iterator(std::end  (sortedList)) };

    std::random_shuffle(tempVector.begin(), tempVector.end());

    sortedList.clear();

    std::copy(tempVector.begin(), tempVector.end(), std::back_inserter(sortedList));
}


Dataset::Dataset(string folderUrl)
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
    // Training pairs
    randomizeList(listPersons);

    // Positives sample

    // For each person...
    for(Person &iter : listPersons) // TODO: Not browse the entire list
    {
        // ... select some random pairs
        cout << iter.getName() << endl;
        for(int i = 0 ; i < 10 ; ++i) // TODO: Replace the arbitrary choosen number
        {
            int number1 = std::rand() % iter.getListImagesId().size();
            int number2 = std::rand() % iter.getListImagesId().size();

            // TODO: Check that the couple has not been selected yet
            if(number1 != number2)
            {
                positiveSamples.push_back(pair<string, string> (iter.getListImagesId().at(number1),
                                                                iter.getListImagesId().at(number2)));
            }
        }
    }

    for (auto iter : positiveSamples)
    {
        cout << iter.first << " " << iter.second << endl;
    }

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

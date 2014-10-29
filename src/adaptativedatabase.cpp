#include "adaptativedatabase.h"

#include <iostream>
#include <fstream>
#include <ctime>

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
}

void AdaptativeDatabase::main()
{
    // Process:

    // Read/load the new sequence
}

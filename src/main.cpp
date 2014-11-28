#include <iostream>

#include "dataset.h"
#include "adaptativedatabase.h"

using namespace std;



int main()
{
    cout << "_________________________________" << endl;
    cout << "Loading dataset..." << endl;

    Dataset dataset("/home/etienne/__A__/Data/Dataset/Set_2/");

    cout << "_________________________________" << endl;
    cout << "Sample selection..." << endl;

    // Choose randomly the training/testing sets
    dataset.selectPairs();

    cout << "_________________________________" << endl;
    cout << "Features extractions..." << endl;

    // Extract features vector for each pairs
    dataset.computeFeatures();

    cout << "_________________________________" << endl;
    cout << "Training..." << endl;

    // Machine learing algorithm
    dataset.train();

    cout << "_________________________________" << endl;
    cout << "Testing..." << endl;

    // Test and plot results
    dataset.test();

    /*cout << "Launch..." << endl;
    AdaptativeDatabase adaptativeDb("/home/etienne/__A__/Data/Dataset/Set_2/");
    adaptativeDb.main();
    adaptativeDb.plotEvaluation();
    cout << "... End of processing" << endl;*/

    return 0;
}

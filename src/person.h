#ifndef PERSON_H
#define PERSON_H

#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Person
{
public:
    Person(const string &namePerson);

    string getName() const;

    void addImageId(const string &newId);
    vector<string> getListImagesId() const;

private:
    string name;
    vector<string> listImagesId;
};

#endif // PERSON_H

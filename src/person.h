#ifndef PERSON_H
#define PERSON_H

#include <iostream>
#include <list>
#include <string>

using namespace std;

class Person
{
public:
    Person(const string &namePerson);

    string getName() const;

    void addImageId(const string &newId);

private:
    string name;
    list<string> listImagesId;
};

#endif // PERSON_H

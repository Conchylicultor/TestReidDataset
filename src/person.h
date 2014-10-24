#ifndef PERSON_H
#define PERSON_H

#include <iostream>
#include <list>
#include <string>

using namespace std;

class Person
{
public:
    Person();

private:
    string name;
    list<string> listImagesId;
};

#endif // PERSON_H
